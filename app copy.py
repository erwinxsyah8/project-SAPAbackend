import os
import io
import torch
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from torch import nn

# =====================================================
# CONFIG
# =====================================================
MODEL_DIR = "./model"
OUTPUT_DIR = "./outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism"
]

# =====================================================
# FASTAPI INIT
# =====================================================
app = FastAPI(
    title="OCEAN Personality Regression API",
    description="Predict Big Five Personality Traits (Likert 0–5)",
    version="1.1.0"
)

# =====================================================
# MODEL DEFINITION (SAMA DENGAN TRAINING)
# =====================================================
class TinyBERTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_DIR)
        self.regressor = nn.Sequential(
            nn.Linear(312, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]
        return self.regressor(cls)

# =====================================================
# LOAD MODEL
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = TinyBERTRegressor().to(DEVICE)
model.regressor.load_state_dict(
    torch.load(f"{MODEL_DIR}/regressor.pt", map_location=DEVICE)
)
model.eval()

print("✅ Model loaded on", DEVICE)

# =====================================================
# SCHEMAS
# =====================================================
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

# =====================================================
# CORE PREDICTION FUNCTION
# =====================================================
def predict_texts(texts: List[str]) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    preds = outputs.cpu().numpy()
    return np.clip(preds, 0.0, 5.0)

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def root():
    return {
        "status": "OK",
        "model": "TinyBERT OCEAN Regression",
        "scale": "Likert 0–5"
    }

# -----------------------------------------------------
# SINGLE TEXT
# -----------------------------------------------------
@app.post("/predict")
def predict_single(data: TextInput):
    pred = predict_texts([data.text])[0]

    return dict(zip(LABELS, map(lambda x: round(float(x), 3), pred)))

# -----------------------------------------------------
# BATCH TEXT
# -----------------------------------------------------
@app.post("/predict-batch")
def predict_batch(data: BatchInput):
    preds = predict_texts(data.texts)

    results = []
    for p in preds:
        results.append(dict(zip(
            LABELS,
            map(lambda x: round(float(x), 3), p)
        )))

    return {
        "count": len(results),
        "results": results
    }

# -----------------------------------------------------
# FILE UPLOAD (CSV / XLSX)
# -----------------------------------------------------
@app.post("/predict-file")
def predict_file(
    file: UploadFile = File(...),
    text_column: str = "text"
):
    if not file.filename.endswith((".xlsx", ".csv")):
        raise HTTPException(
            status_code=400,
            detail="File harus CSV atau XLSX"
        )

    # Read file
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        df = pd.read_excel(file.file)

    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom '{text_column}' tidak ditemukan"
        )

    texts = df[text_column].astype(str).tolist()
    preds = predict_texts(texts)

    # Append predictions
    for i, label in enumerate(LABELS):
        df[label] = preds[:, i].round(3)

    # Save output
    output_path = os.path.join(
        OUTPUT_DIR,
        f"prediction_{file.filename.replace('.', '_')}.xlsx"
    )
    df.to_excel(output_path, index=False)

    return {
        "status": "success",
        "rows": len(df),
        "output_file": output_path
    }
