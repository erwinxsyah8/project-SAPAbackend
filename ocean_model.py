import torch
import torch.nn as nn
from transformers import AutoModel


class OceanRegressor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(0.3)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, input_ids, attention_mask, return_logits=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0]  # CLS token
        cls_embedding = self.dropout(cls_embedding)

        logits = self.regressor(cls_embedding)

        if return_logits:
            return logits

        # Normalize to 0–1 for OCEAN regression
        return torch.sigmoid(logits)
