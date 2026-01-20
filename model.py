# app/model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download

HF_REPO = "sapadev13/sapa_ocean_id"
DEVICE = "cpu"

class OceanModel(nn.Module):
    def __init__(self, encoder, lexical_size):
        super().__init__()
        hidden = encoder.config.hidden_size
        self.encoder = encoder
        self.fc = nn.Linear(hidden + lexical_size, 5)

    def forward(self, input_ids, attention_mask, lexical):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        x = torch.cat([cls, lexical], dim=1)
        return self.fc(x)


# ⬇️⬇️⬇️ INI WAJIB ADA & DI LEVEL FILE (BUKAN DI DALAM CLASS)
def load_model(lexical_size: int):
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    encoder = AutoModel.from_pretrained(HF_REPO)

    model = OceanModel(encoder, lexical_size)

    state_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="pytorch_model.bin"
    )
    state_dict = torch.load(state_path, map_location=DEVICE)

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    return model, tokenizer
