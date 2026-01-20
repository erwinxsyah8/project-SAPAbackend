import torch
from .model import load_model
from .lexical import (
    build_lexical_vector_with_analysis,
    adjust_ocean_by_keywords,
    apply_emotional_keyword_adjustment,
    generate_persona_profile,
    generate_explanation_suggestion_super,
    highlight_keywords_in_text
)

DEVICE = "cpu"
MAX_LEN = 256

model, tokenizer = load_model(lexical_size=LEXICAL_SIZE)

def run_ocean_pipeline(text: str, username=None):
    lexical, coverage, subtraits, evidence = build_lexical_vector_with_analysis(text)
    lexical = lexical.to(DEVICE)

    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        out = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            lexical
        )

    raw = {
        "O": out[0,0].item(),
        "C": out[0,1].item(),
        "E": out[0,2].item(),
        "A": out[0,3].item(),
        "N": out[0,4].item(),
    }

    dominant, adjusted = adjust_ocean_by_keywords(raw, text)
    adjusted = apply_emotional_keyword_adjustment(text, adjusted)

    explanation, suggestion = generate_explanation_suggestion_super(
        text, adjusted, evidence
    )

    return {
        "username": username,
        "highlighted_text": highlight_keywords_in_text(text, evidence),
        "prediction_adjusted": adjusted,
        "dominant_trait": dominant,
        "personality_profile": generate_persona_profile(adjusted),
        "explanation": explanation,
        "suggestion": suggestion
    }
