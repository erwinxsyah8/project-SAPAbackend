# app/lexical.py

import numpy as np

LEXICAL_DIM = 10

def build_lexical_features(text: str):
    """
    Dummy lexical feature extractor
    (bisa kamu ganti nanti dengan versi lengkap)
    """
    vec = np.zeros(LEXICAL_DIM, dtype="float32")

    text = text.lower()
    vec[0] = len(text)
    vec[1] = text.count("!")
    vec[2] = text.count("?")
    vec[3] = sum(c.isupper() for c in text)
    vec[4] = len(text.split())

    return vec
def build_lexical_vector_with_analysis(text: str):
    """
    Membangun vektor leksikal dan analisis sederhana
    """
    vec = build_lexical_features(text)
    lexical_tensor = torch.tensor(vec).unsqueeze(0)  # Bentuk (1, LEXICAL_DIM)

    # Analisis sederhana (dummy)
    coverage = 0.5  # Misal, 50% kata tercover
    subtraits = {
        "subtrait_1": 0.6,
        "subtrait_2": 0.4
    }
    evidence = {
        "exclamation_count": vec[1],
        "question_count": vec[2]
    }

    return lexical_tensor, coverage, subtraits, evidence
def adjust_ocean_by_keywords(raw_ocean, text):
    """
    Penyesuaian OCEAN berdasarkan kata kunci sederhana
    """
    dominant = max(raw_ocean, key=raw_ocean.get)
    adjusted = raw_ocean.copy()

    if "happy" in text.lower():
        adjusted["E"] += 0.1
    if "sad" in text.lower():
        adjusted["N"] += 0.1

    return dominant, adjusted
def apply_emotional_keyword_adjustment(text, ocean_scores):
    """
    Penyesuaian tambahan berdasarkan kata kunci emosional
    """
    if "angry" in text.lower():
        ocean_scores["A"] += 0.1
    return ocean_scores
def generate_persona_profile(ocean_scores):
    """
    Menghasilkan profil kepribadian berdasarkan skor OCEAN
    """
    profile = {
        "Openness": ocean_scores["O"],
        "Conscientiousness": ocean_scores["C"],
        "Extraversion": ocean_scores["E"],
        "Agreeableness": ocean_scores["A"],
        "Neuroticism": ocean_scores["N"],
    }
    return profile
def generate_explanation_suggestion_super(text, ocean_scores, evidence):
    """
    Menghasilkan penjelasan dan saran berdasarkan skor OCEAN
    """
    explanation = f"The text shows high {max(ocean_scores, key=ocean_scores.get)} traits."
    suggestion = "Consider using more varied vocabulary to enhance openness."
    return explanation, suggestion
def highlight_keywords_in_text(text, evidence):
    """
    Menyoroti kata kunci dalam teks berdasarkan evidence
    """
    highlighted = text
    if evidence.get("exclamation_count", 0) > 0:
        highlighted = highlighted.replace("!", "[!]")
    if evidence.get("question_count", 0) > 0:
        highlighted = highlighted.replace("?", "[?]")
    return highlighted
import torch
LEXICAL_SIZE = 10
# LEXICAL_SIZE harus sesuai dengan dimensi vektor leksikal
# yang dihasilkan oleh fungsi build_lexical_features
# di file ini (LEXICAL_DIM)
# Jadi, pastikan LEXICAL_SIZE = LEXICAL_DIM
# di file ini (LEXICAL_DIM)
LEXICAL_SIZE = LEXICAL_DIM
