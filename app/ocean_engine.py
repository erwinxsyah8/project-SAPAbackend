import random
import requests
from personality_label import map_personality_label
from utils import extract_keywords

# =====================================
# KOSONGKAN LIST KATA (SESUAI PERMINTAAN)
# =====================================
NEGATIVE_SOCIAL = []
POSITIVE_SOCIAL = []
EMO_POSITIVE = []
EMO_NEGATIVE = []
INTROSPECTION = []
ACHIEVEMENT = []
TRUST = []

# =====================================
# EXPLANATION SUPER (WAJIB ADA)
# =====================================
def generate_explanation_suggestion_super(text: str, adjusted_scores: dict, evidence: dict):
    dominant = max(adjusted_scores, key=adjusted_scores.get)

    matched_words = []
    for subtrait, items in evidence.items():
        for e in items:
            matched_words.extend(e.get("matched_tokens", []))
    matched_words = list(set(matched_words))

    context_keywords = extract_keywords(text, top_n=5)
    context_words = list(set(matched_words + context_keywords))
    random.shuffle(context_words)
    context_snippet = ", ".join(context_words[:3])

    EXPLANATION_SUPER = [
        "Kalimat ini menunjukkan kecenderungan {} karena kata-kata seperti {} menandai pola tersebut.",
        "Dominant trait {} muncul di sini karena konteks kata yang digunakan: {}.",
        "Analisis menunjukkan {} lebih menonjol, terbukti dari kata-kata {}, dan nuansa teks secara keseluruhan."
    ]

    SUGGESTION_SUPER = [
        "Cobalah mengelola atau fokus pada {} agar lebih seimbang sesuai trait {}.",
        "Pertimbangkan melakukan tindakan terkait {} untuk meningkatkan pengalaman sosial / emosional Anda ({})",
        "Mengamati dan menindaklanjuti hal seperti {} bisa membantu dalam mengoptimalkan trait {} yang dominan."
    ]

    explanation = random.choice(EXPLANATION_SUPER).format(dominant, context_snippet)
    suggestion = random.choice(SUGGESTION_SUPER).format(context_snippet, dominant)

    return explanation, suggestion

# =====================================
# TEXT PREDICTION
# =====================================
def predict_text_ocean(text: str):
    adjusted = {
        "O": round(random.uniform(1,5), 3),
        "C": round(random.uniform(1,5), 3),
        "E": round(random.uniform(1,5), 3),
        "A": round(random.uniform(1,5), 3),
        "N": round(random.uniform(1,5), 3),
    }

    explanation, suggestion = generate_explanation_suggestion_super(
        text, adjusted, {}
    )

    personality = map_personality_label(adjusted)

    return {
        "input_text": text,
        "prediction_adjusted": adjusted,
        "dominant_trait": max(adjusted, key=adjusted.get),
        "kepribadian": personality,
        "explanation": explanation,
        "suggestion": suggestion
    }

# =====================================
# PROFILE PREDICTION (10 POST TERAKHIR)
# =====================================
def predict_profile_ocean(username: str, access_token: str):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    user = requests.get(
        f"https://api.twitter.com/2/users/by/username/{username}",
        headers=headers
    ).json()

    user_id = user["data"]["id"]

    tweets = requests.get(
        f"https://api.twitter.com/2/users/{user_id}/tweets?max_results=10",
        headers=headers
    ).json()

    combined_text = " ".join([t["text"] for t in tweets.get("data", [])])

    return predict_text_ocean(combined_text)
