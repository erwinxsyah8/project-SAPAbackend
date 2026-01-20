import torch
import numpy as np
from .model import model, tokenizer, device, TRAITS

@torch.no_grad()
def predict(texts: list[str]):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ===== FORWARD =====
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    # 🔴 FIX UTAMA DI SINI
    logits = outputs.logits            # Tensor [B, 5]
    logits = torch.clamp(logits, 0.0, 5.0)

    scores = logits.cpu().numpy()

    results = []

    for i, row in enumerate(scores):
        row = row.astype(float)

        # ===== uncertainty & confidence =====
        variance = float(np.var(row))
        uncertainty = np.clip(variance / 2.5, 0.0, 1.0)

        spread = float(np.max(row) - np.min(row))
        confidence = np.clip((spread / 5.0) * (1.0 - uncertainty), 0.0, 1.0)

        item = {
            "text": texts[i],
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
        }

        for trait, val in zip(TRAITS, row):
            item[trait] = round(val, 3)

        results.append(item)

    return results
