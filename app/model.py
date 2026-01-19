import torch
import torch.nn as nn
from transformers import AutoModel

STUDENT_NAME = "huawei-noah/TinyBERT_General_4L_312D"
LABEL_COLS = ["O", "C", "E", "A", "N"]

class TinyBERT_TFIDF(nn.Module):
    def __init__(self, tfidf_dim):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(STUDENT_NAME)

        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.regressor = nn.Sequential(
            nn.Linear(312 + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, input_ids, attention_mask, tfidf):
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = enc.last_hidden_state[:, 0]
        tfidf_feat = self.tfidf_proj(tfidf.float())
        fused = torch.cat([cls, tfidf_feat], dim=1)
        return self.regressor(fused)
