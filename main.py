import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import tweepy
import math
import os

import matplotlib.pyplot as plt
import base64, hashlib
import pandas as pd
from fastapi import UploadFile, File
from dotenv import load_dotenv
from io import BytesIO
from tweepy import OAuth2UserHandler
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from oauthlib.common import generate_token
from requests_oauthlib import OAuth2Session
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from collections import defaultdict, Counter
from app.logger_setup import logger
from pathlib import Path


os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

load_dotenv()
logger.info("FastAPI app starting...")

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
if not TWITTER_API_KEY or not TWITTER_API_SECRET:
    raise RuntimeError("TWITTER_API_KEY or TWITTER_API_SECRET not set in .env")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")
if not TWITTER_CLIENT_ID:
    raise RuntimeError("TWITTER_CLIENT_ID is not set in .env")
# TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET")
TWITTER_REDIRECT_URI = "http://localhost:8000/auth/twitter/callback"
TWITTER_SCOPES = ["tweet.read", "users.read", "offline.access"]
# ==========================
# CONFIG
# ==========================
BASE_DIR = Path(__file__).resolve().parent

ONTOLOGY_CSV = BASE_DIR / "ontology_clean.csv"
ONTOLOGY_EMB = BASE_DIR / "ontology_embeddings.pt"

HF_REPO = "sapadev13/sapa_ocean_id"
DEVICE = "cpu"
MAX_LEN = 256

# ==========================
# LOAD ONTOLOGY CSV
# ==========================ontology_df = None
SUBTRAITS = None
LEXICAL_SIZE = None
subtrait2id = None
LEXICON = None
ONT_EMBEDDINGS = None
ONT_META = None

# ==========================
# MODEL DEFINITION
# ==========================
class OceanModel(nn.Module):
    def __init__(self, encoder, lexical_size):
        super().__init__()

        if lexical_size is None:
            raise ValueError("lexical_size tidak boleh None")

        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.fc = nn.Linear(hidden + lexical_size, 5)

    def forward(self, input_ids, attention_mask, lexical):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        x = torch.cat([cls, lexical], dim=1)
        return self.fc(x)

# ==========================
# LOAD MODEL FROM HF
# ==========================

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(
    title="SAPA OCEAN API",
    description="Ontology-aware Indonesian Personality Prediction",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sapadev.id"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # ⬅️ INI PENTING (OPTIONS termasuk)
    allow_headers=["*"],
)
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-secret-CHANGE-ME")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="lax",
    https_only=False
)
def get_twitter_client(access_token, access_token_secret):
    client = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_SECRET,
    access_token=access_token,
    access_token_secret=None  # Optional, kalau OAuth 2.0 User Context
    )
    return client
class TextInput(BaseModel):
    text: str

# ==========================
# LEXICAL ENGINE
# ==========================
def build_lexical_vector_with_analysis(text: str):
    vec = torch.zeros(LEXICAL_SIZE)
    tokens = re.findall(r"\w+", text.lower())
    token_set = set(tokens)

    matched_tokens = set()
    subtrait_scores = defaultdict(float)
    evidence = defaultdict(list)

    for subtrait, patterns in LEXICON.items():
        sid = subtrait2id[subtrait]
        for p in patterns:
            overlap = p["tokens"] & token_set
            if not overlap:
                continue

            ratio = len(overlap) / len(p["tokens"])
            if ratio == 1.0:
                score = 2.0 * p["strength"]
            elif ratio >= 0.5:
                score = 0.5 * p["strength"]
            else:
                continue

            vec[sid] += score
            subtrait_scores[subtrait] += score
            matched_tokens |= overlap

            evidence[subtrait].append({
                "lexeme": p["lexeme"],
                "matched_tokens": list(overlap),
                "score": round(score, 3)
            })

    vec = torch.log1p(vec)
    if vec.sum() > 0:
        vec = vec / vec.sum()

    coverage = len(matched_tokens) / max(len(token_set), 1)
    return vec.unsqueeze(0), round(coverage * 100, 2), dict(sorted(subtrait_scores.items(), key=lambda x: -x[1])), evidence

# ==========================
# ONTOLOGY EMBEDDING EXPANSION
# ==========================
def expand_ontology_candidates(text: str, top_k=5, threshold=0.7):
    tokens = text.lower().split()
    text_vecs = []

    for i, meta in enumerate(ONT_META):
        if meta["lexeme"] in tokens:
            text_vecs.append(ONT_EMBEDDINGS[i])

    if not text_vecs:
        text_vecs = [np.zeros(ONT_EMBEDDINGS.shape[1])]

    text_vec = np.mean(text_vecs, axis=0)
    candidates = []

    for i, meta in enumerate(ONT_META):
        lex_vec = ONT_EMBEDDINGS[i]
        sim = np.dot(text_vec, lex_vec) / (np.linalg.norm(text_vec) * np.linalg.norm(lex_vec) + 1e-8)
        if sim >= threshold:

            candidates.append({
                "candidate_from": meta["lexeme"],
                "suggested_subtrait": meta["sub_trait"],
                "similarity": round(float(sim), 3)
            })

    return sorted(candidates, key=lambda x: x["similarity"])[:top_k]
# BASE_DIR = folder tempat script ini berada
BASE_DIR = Path(__file__).parent  # __file__ = lokasi script ini

# Path lengkap ke file Excel
excel_path = BASE_DIR / "keywords_traits.xlsx"

print("Mencoba membuka file:", excel_path)  # Debug
df_keywords = pd.read_excel(excel_path)

TRAIT_KEYWORDS = {}
for _, row in df_keywords.iterrows():
    trait = str(row["Trait / Kategori"]).strip()  # pastikan string tanpa spasi
    word = str(row["Keyword / Phrase"]).strip().lower()
    if trait not in TRAIT_KEYWORDS:
        TRAIT_KEYWORDS[trait] = []
    TRAIT_KEYWORDS[trait].append(word)

# ==========================
# CEK JUMLAH KATA PER TRAIT
# ==========================
for trait, words in TRAIT_KEYWORDS.items():
    print(f"{trait}: {len(words)} kata")

# ==========================
# ASSIGN KE VARIABEL GLOBAL (LIST)
# ==========================
NEGATIVE_SOCIAL = TRAIT_KEYWORDS.get("NEGATIVE_SOCIAL", [])
POSITIVE_SOCIAL = TRAIT_KEYWORDS.get("POSITIVE_SOCIAL", [])
EMO_POSITIVE = TRAIT_KEYWORDS.get("EMO_POSITIVE", [])
EMO_NEGATIVE = TRAIT_KEYWORDS.get("EMO_NEGATIVE", [])
INTROSPECTION = TRAIT_KEYWORDS.get("INTROSPECTION", [])
ACHIEVEMENT = TRAIT_KEYWORDS.get("ACHIEVEMENT", [])
CREATIVE_DISCUSSION_A = TRAIT_KEYWORDS.get("CREATIVE_DISCUSSION_A", [])
TRUST = TRAIT_KEYWORDS.get("TRUST", [])
RELATIONSHIP_AFFECTION = TRAIT_KEYWORDS.get("RELATIONSHIP_AFFECTION", [])
COLLABORATION = TRAIT_KEYWORDS.get("COLLABORATION", [])
ANGER_EMO = TRAIT_KEYWORDS.get("ANGER_EMO", [])
SAD_EMO = TRAIT_KEYWORDS.get("SAD_EMO", [])
ANXIETY_EMO = TRAIT_KEYWORDS.get("ANXIETY_EMO", [])
EXTREME_NEGATIVE = TRAIT_KEYWORDS.get("EXTREME_NEGATIVE", [])
DISCIPLINE_C = TRAIT_KEYWORDS.get("DISCIPLINE_C", [])
EXTRAVERSION_E = TRAIT_KEYWORDS.get("EXTRAVERSION_E", [])
E_SOCIAL_DEPENDENCY = TRAIT_KEYWORDS.get("E_SOCIAL_DEPENDENCY", [])
EMPATHY_HARMONY_A = TRAIT_KEYWORDS.get("EMPATHY_HARMONY_A", [])
MENTAL_UNSTABLE_N = TRAIT_KEYWORDS.get("MENTAL_UNSTABLE_N", [])

# ==========================
# OCEAN ADJUSTMENT REFINED
# ==========================
from collections import Counter
import re
def adjust_ocean_by_keywords(scores: dict, text: str):
    adjusted = scores.copy()
    counter = Counter(re.findall(r'\w+', text.lower()))

    # NEGATIVE_SOCIAL → menurunkan E, menaikkan N, sedikit turunkan A
    for word in NEGATIVE_SOCIAL:
        if word in counter:
            f = counter[word]
            adjusted["E"] -= 0.2 * f
            adjusted["N"] += 0.5 * f
            adjusted["A"] -= 0.1 * f

    # POSITIVE_SOCIAL → menaikkan E & A, sedikit menurunkan N jika sangat negatif
    for word in POSITIVE_SOCIAL:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.4 * f
            adjusted["E"] += 0.3 * f
            adjusted["N"] -= 0.05 * f

    # EMO_POSITIVE → tingkatkan E & O (turunkan O 0.85x)
    for word in EMO_POSITIVE:
        if word in counter:
            f = counter[word]
            adjusted["E"] += 0.3 * f
            adjusted["O"] += 0.15 * f * 0.85

    # EMO_NEGATIVE → tingkatkan N & sedikit O (turunkan O 0.85x)
    for word in EMO_NEGATIVE:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.35 * f
            adjusted["O"] += 0.1 * f * 0.85

    # INTROSPECTION → tingkatkan O (turunkan O 0.80x)
    for word in INTROSPECTION:
        if word in counter:
            f = counter[word]
            if not any(n in text.lower() for n in MENTAL_UNSTABLE_N):
                adjusted["O"] += 0.35 * f * 0.8

    # ACHIEVEMENT → tingkatkan C
    for word in ACHIEVEMENT:
        if word in counter:
            f = counter[word]
            adjusted["C"] += 0.5 * f

    # TRUST → tingkatkan A
    for word in TRUST:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.5 * f

    # RELATIONSHIP_AFFECTION → naikkan A, sedikit turunkan N
    for word in RELATIONSHIP_AFFECTION:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.6 * f
            adjusted["N"] -= 0.1 * f

    for word in COLLABORATION:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.9 * f
            adjusted["E"] += 0.7 * f
            adjusted["N"] -= 0.05 * f

    # EXTREME_NEGATIVE → N naik lebih tinggi, E turun sedikit
    for phrase in EXTREME_NEGATIVE:
        if phrase in text.lower():
            adjusted["N"] += 1.0
            adjusted["E"] -= 0.2
            adjusted["EXTREME_ALERT"] = True

    # CREATIVE DISCUSSION (A-dominant) → turunkan O 0.85x
    for phrase in CREATIVE_DISCUSSION_A:
        if phrase in text.lower():
            adjusted["A"] -= 0.3
            adjusted["O"] += 0.4 * 0.85
            adjusted["E"] += 0.2

    for word in DISCIPLINE_C:
        if word in counter:
            f = counter[word]
            adjusted["C"] += 1.2 * f
            adjusted["O"] -= 0.4

    for word in EXTRAVERSION_E:
        if word in counter:
            f = counter[word]
            adjusted["E"] += 0.7 * f
            adjusted["A"] += 0.3 * f
            adjusted["N"] -= 0.05 * f

    for phrase in E_SOCIAL_DEPENDENCY + ACHIEVEMENT + EXTRAVERSION_E:
        if phrase.lower() in text.lower():
            adjusted["E"] += 0.6
            adjusted["A"] -= 0.7
            adjusted["O"] -= 0.8

    for word in EMPATHY_HARMONY_A:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.7 * f
            adjusted["N"] -= 0.1 * f

    # MENTAL UNSTABLE / OVERTHINKING → turunkan O 0.75x
    for phrase in MENTAL_UNSTABLE_N:
        if phrase in text.lower():
            adjusted["N"] += 8.0
            adjusted["E"] -= 0.2
            adjusted["O"] += -0.1 * 0.75

    # Clamp ke skala 1–5
    for k in adjusted:
        adjusted[k] = round(min(5.0, max(1.0, adjusted[k])), 3)

    return max(adjusted, key=adjusted.get), adjusted

# ==========================
# KEYWORD → OCEAN MAPPING
# ==========================
KEYWORD_TRAIT_MAP = {
    "NEGATIVE_SOCIAL": {"E": -0.2, "N": 0.5, "A": -0.1},
    "POSITIVE_SOCIAL": {"A": 0.3, "E": 0.6, "N": -0.05},
    "EMO_POSITIVE": {"E": 0.3, "O": 0.15},
    "EMO_NEGATIVE": {"N": 0.35, "O": 0.1},
    "INTROSPECTION": {"O": 0.35},
    "ACHIEVEMENT": {"C": 0.5},
    "CREATIVE_DISCUSSION_A": {"A": 0.8, "E": 0.4, "C": 0.2, "N": -0.05},
    "TRUST": {"A": 0.5},
    "RELATIONSHIP_AFFECTION": {"A": 0.6, "N": -0.1},
    "COLLABORATION": {"A": 0.8, "E": 0.5, "C": 0.2, "N": -0.05},
    "ANGER_EMO": {"N": 0.5},
    "SAD_EMO": {"N": 0.3, "O": 0.05},
    "ANXIETY_EMO": {"N": 0.35, "E": -0.1},
    "EXTREME_NEGATIVE": {"N": 1.0, "E": -0.2},
    "DISCIPLINE_C": {"C": 1.2, "O": -0.4},
    "EXTRAVERSION_E": {"E": 0.6, "A": -0.7, "O": -0.8},
    "E_SOCIAL_DEPENDENCY": {"E": 0.6, "A": 0.4, "N": -0.1},
    "EMPATHY_HARMONY_A": {"A": 0.7, "N": -0.1},
    "MENTAL_UNSTABLE_N": {"N": 8.0, "E": -0.2, "O": -0.1},

}

# ==========================
# EMOTIONAL KEYWORD ADJUSTMENT REFINED
# ==========================
def apply_emotional_keyword_adjustment(text: str, scores: dict, o_reduce: float = 0.5):
    """
    Adjust OCEAN scores based on emotional keywords.
    o_reduce: angka positif, O akan dikurangi o_reduce poin dari nilai sebelumnya
    """
    adjusted = scores.copy()
    text_lower = text.lower()
    tokens = re.findall(r'\w+', text_lower)
    counter = Counter(tokens)

    # ==========================
    # KATA EMOSI NEGATIF → Naikkan N
    # ==========================
    for word in ANGER_EMO + SAD_EMO + ANXIETY_EMO + EMO_NEGATIVE + NEGATIVE_SOCIAL + MENTAL_UNSTABLE_N:
        if word in tokens:
            f = counter[word]
            adjusted["N"] += 0.3 * f
            if word in ANXIETY_EMO:
                adjusted["E"] -= 0.1 * f
            if word in SAD_EMO:
                adjusted["O"] += 0.05 * f

    # ==========================
    # KATA EMOSI POSITIF → Naikkan A/E/O & turunkan N sedikit
    # ==========================
    for word in EMO_POSITIVE + POSITIVE_SOCIAL + COLLABORATION + ACHIEVEMENT + TRUST + RELATIONSHIP_AFFECTION + EXTRAVERSION_E + E_SOCIAL_DEPENDENCY + EMPATHY_HARMONY_A + CREATIVE_DISCUSSION_A + DISCIPLINE_C + INTROSPECTION:
        if word in tokens:
            f = counter[word]
            if word in EMO_POSITIVE + POSITIVE_SOCIAL + COLLABORATION:
                adjusted["A"] += 0.35 * f
                adjusted["N"] -= 0.1 * f
            if word in EXTRAVERSION_E + E_SOCIAL_DEPENDENCY + CREATIVE_DISCUSSION_A:
                adjusted["O"] += 0.55 * f
                adjusted["E"] += 0.35 * f
                adjusted["A"] -= 0.35 * f
            if word in ACHIEVEMENT + DISCIPLINE_C + INTROSPECTION + TRUST + EMPATHY_HARMONY_A:
                adjusted["E"] += 0.45 * f
                adjusted["O"] += 0.45 * f
                adjusted["N"] -= 0.45 * f

    # ==========================
    # Kata spesifik override ringan → turunkan N
    # ==========================
    light_positive_words = ["suka", "senang", "belajar", "bahagia", "ceria", "menyenangkan"]
    for word in light_positive_words:
        if word in tokens:
            adjusted["N"] = max(1.0, adjusted["N"] - 0.5)

    # ==========================
    # TURUNKAN Openness dengan jumlah tetap
    # ==========================
    adjusted["O"] = max(1.0, round(adjusted["O"] - o_reduce, 3))

    # ==========================
    # Clamp trait lain ke skala 1–5
    # ==========================
    for k in ["C", "E", "A", "N"]:
        adjusted[k] = round(min(5.0, max(1.0, adjusted[k])), 3)

    return adjusted


# ==========================
# DOMINANT TRAIT SEDERHANA
# ==========================
def determine_dominant_trait(scores, text):
    """
    Pilih trait dominan berdasarkan skor tertinggi,
    tapi override jika kata positif dominan.
    """
    text_lower = text.lower()
    tokens = re.findall(r'\w+', text_lower)

    # Hitung kata positif
    social_hits = sum(1 for w in POSITIVE_SOCIAL + COLLABORATION if w in tokens)
    emo_hits = sum(1 for w in EMO_POSITIVE if w in tokens)
    achievement_hits = sum(1 for w in ACHIEVEMENT + DISCIPLINE_C if w in tokens)
    light_hits = sum(1 for w in ["suka", "senang", "belajar", "bahagia", "ceria"] if w in tokens)

    if social_hits >= 1:
        return "A"
    if emo_hits >= 1:
        return "E"
    if achievement_hits >= 1 or light_hits >= 1:
        return "O"

    # Default: trait tertinggi
    return max(scores, key=scores.get)
import re
from collections import Counter

# ==========================
# HIGHLIGHT YANG AKURAT
# ==========================
def highlight_keywords_in_text(text: str, evidence: dict):
    """
    Memberikan highlight kata-kata yang ditemukan di evidence.
    """
    tokens = re.findall(r'\w+|\W+', text)
    highlights = set()

    # Ambil semua kata bukti dari evidence
    for key, items in evidence.items():
        for e in items:
            matched = e.get("matched_tokens", [])
            highlights.update([t.lower() for t in matched])

    # Bangun teks dengan <mark>
    result = "".join(f"<mark>{t}</mark>" if t.lower() in highlights else t for t in tokens)
    return result


# ==========================
# SUPER EXPLANATION YANG DISERDERHANAKAN
# ==========================
def extract_keywords(text, top_n=5):
    """
    Ambil kata paling sering muncul sebagai highlight
    """
    return [w for w, _ in Counter(re.findall(r'\w+', text.lower())).most_common(top_n)]

def generate_explanation_suggestion_super(text, adjusted, evidence):
    """
    Buat penjelasan dan saran berdasarkan adjusted scores & evidence
    """
    dominant = max(adjusted, key=adjusted.get)
    words = extract_keywords(text)
    snippet = ", ".join(words[:3])

    # Peringatan jika ada EXTREME_ALERT
    if adjusted.get("EXTREME_ALERT", 0) > 0:
        explanation = (
            f"⚠ Kalimat ini mengandung indikasi emosional ekstrem / risiko tinggi. "
            f"Kecenderungan trait {dominant} tetap terlihat. "
            f"Kata-kata seperti {snippet} menunjukkan hal tersebut."
        )
        suggestion = (
            f"Sangat disarankan untuk memberikan perhatian atau dukungan psikologis segera. "
            f"Memantau kata-kata seperti {snippet} dapat membantu mengurangi risiko."
        )
    else:
        explanation = (
            f"Kalimat ini menunjukkan kecenderungan {dominant} karena kata-kata seperti {snippet} menandai pola tersebut."
        )
        suggestion = (
            f"Mengamati hal seperti {snippet} dapat membantu memahami dan mengoptimalkan trait {dominant}."
        )

    return explanation, suggestion


# ==========================
# DOMINANT CONTEXTUAL YANG DISERDERHANAKAN
# ==========================
def determine_dominant_contextual(adjusted, evidence):
    """
    Pilih trait dominan berdasarkan skor tertinggi, tapi override jika ada bukti sosial/emosi positif
    """
    # Hitung jumlah bukti sosial / emosional
    social_hits = len(evidence.get("POSITIVE_SOCIAL", [])) + len(evidence.get("COLLABORATION", []))
    emo_hits = len(evidence.get("EMO_POSITIVE", []))

    if social_hits >= 2:
        return "A"
    if emo_hits >= 2:
        return "E"

    # Default: trait tertinggi
    sorted_traits = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
    return sorted_traits[0][0]  # trait dengan skor tertinggi
PERSONA_RULES = [
    # ===== EMOSI NEGATIF =====
    ("Cemas & Pikiran Tidak Tenang", lambda s: s["N"] >= 3.6, "mudah gugup, sulit tenang, sering pikiran negatif"),
    ("Overthinking Emosional", lambda s: s["N"] >= 3.5 and s["O"] <= 3.2, "pikiran sulit dikendalikan, sering cemas"),
    ("Rentan Stres", lambda s: s["N"] >= 3.4 and s["C"] <= 3.0, "mudah tertekan, butuh manajemen stres"),
    ("Tempramental", lambda s: s["N"] >= 4.0, "cepat marah, impulsif, reaktif"),
    ("Cemas & Overthinking", lambda s: s["N"] >= 3.5 and s["E"] <= 3.0, "mudah khawatir dan gelisah"),
    ("Melankolis", lambda s: s["N"] >= 3.2 and s["O"] >= 3.0 and s["E"] <= 3.2, "sering merenung, introspektif"),

    # ===== EMOSI POSITIF =====
    ("Empatik & Penjaga Harmoni", lambda s: s["A"] >= 3.7, "peduli, mudah memahami orang lain"),
    ("Pendamai Alami", lambda s: s["A"] >= 3.6 and s["N"] <= 3.2, "menghindari konflik, menciptakan damai"),
    ("Kolaborator Hangat", lambda s: s["A"] >= 3.5 and s["E"] >= 3.0, "mudah bekerja sama dan suportif"),
    ("Sosial Aktif", lambda s: s["E"] >= 3.6, "energik dan aktif berinteraksi"),
    ("Penggerak Diskusi", lambda s: s["E"] >= 3.5 and s["O"] >= 3.0, "suka berdiskusi dan memancing ide"),
    ("Ekstrovert Sosial", lambda s: s["E"] >= 3.8 and s["N"] <= 3.0, "percaya diri, mudah bergaul"),
    ("Romantis", lambda s: s["A"] >= 3.4 and s["A"] >= s["O"] + 0.2, "hangat dan berorientasi hubungan"),
    ("Ramah Sosial", lambda s: s["E"] >= 3.5 and s["A"] >= 3.2, "ceria, mudah bergaul"),
    ("Empatik", lambda s: s["A"] >= 3.5 and s["N"] <= 3.2, "peduli dan suportif"),
    ("Visioner Kreatif", lambda s: s["O"] >= 3.7 and s["O"] >= s["A"] + 0.2, "imajinatif, terbuka terhadap ide baru"),
    ("Inovator", lambda s: s["O"] >= 3.5 and s["C"] >= 3.0 and s["E"] >= 3.0, "kreatif dan berpikir out-of-the-box"),

    # ===== PENCAPAIAN & DISCIPLIN =====
    ("Perfeksionis", lambda s: s["C"] >= 3.6 and s["C"] >= s["N"] + 0.2, "terstruktur, disiplin, berorientasi pencapaian"),
    ("Ambisius", lambda s: s["C"] >= 3.5 and s["O"] >= 3.5, "proaktif dan berinisiatif"),
    ("Gigih & Persisten", lambda s: s["C"] >= 3.4 and s["N"] <= 3.2, "konsisten dan berdedikasi"),
    ("Pragmatis", lambda s: s["C"] >= 3.2 and s["E"] >= 3.2, "praktis dan fokus pada hasil"),

    # ===== KOLABORATOR =====
    ("Kolaborator", lambda s: s["A"] >= 3.5 and s["E"] >= 3.2 and s["C"] >= 3.0, "mampu bekerja sama dan membangun harmoni"),
    ("Mediator", lambda s: s["A"] >= 3.3 and s["N"] <= 3.2 and s["E"] >= 3.0, "tenang dan diplomatis"),
    ("Pemimpin Visioner", lambda s: s["O"] >= 3.6 and s["C"] >= 3.5 and s["E"] >= 3.2, "memimpin tim dan strategis"),

    # ===== KEPRIBADIAN SEIMBANG =====
    ("Seimbang", lambda s: all(2.8 <= s[k] <= 3.5 for k in ["O","C","E","A","N"]), "adaptif, fleksibel, tidak ekstrem"),
]


# ==========================
# GLOBAL CONCLUSION
# ==========================
def generate_global_conclusion(avg, dominant):
    O, C, E, A, N = avg["O"], avg["C"], avg["E"], avg["A"], avg["N"]

    conclusion = (
        f"Secara keseluruhan, trait kepribadian paling dominan adalah {dominant}. Individu ini cenderung "
    )

    if dominant == "O":
        conclusion += "kreatif, reflektif, dan terbuka terhadap ide baru."
    elif dominant == "C":
        conclusion += "terstruktur, disiplin, konsisten, dan bertanggung jawab."
    elif dominant == "E":
        conclusion += "aktif secara sosial, komunikatif, dan energik."
    elif dominant == "A":
        conclusion += "kooperatif, empatik, dan menjaga keharmonisan sosial."
    elif dominant == "N":
        conclusion += "sensitif terhadap tekanan emosional dan mudah mengalami stres."

    # Insight tambahan dari Neuroticism
    if N < 0.35:
        conclusion += " Tingkat kestabilan emosi tergolong baik."
    elif N > 0.6:
        conclusion += " Terdapat kecenderungan emosi negatif yang cukup tinggi."

    # ================= SARAN =================
    suggestion = "Disarankan untuk "
    if dominant == "C":
        suggestion += "memanfaatkan kemampuan perencanaan dan kedisiplinan, tetapi tetap fleksibel."
    elif dominant == "O":
        suggestion += "menyalurkan kreativitas ke aktivitas produktif dan eksplorasi ide."
    elif dominant == "E":
        suggestion += "mengoptimalkan kemampuan komunikasi, kepemimpinan, dan refleksi diri."
    elif dominant == "A":
        suggestion += "mempertahankan empati sambil belajar bersikap tegas saat dibutuhkan."
    elif dominant == "N":
        suggestion += "melatih regulasi emosi melalui manajemen stres, mindfulness, atau journaling rutin."

    return conclusion, suggestion
OCEAN_COLORS = {
    "O": "#6366F1",  # Indigo – Openness
    "C": "#22C55E",  # Green – Conscientiousness
    "E": "#F59E0B",  # Amber – Extraversion
    "A": "#3B82F6",  # Blue – Agreeableness
    "N": "#EF4444"   # Red – Neuroticism
}

def ocean_to_bar_chart(avg_ocean):
    """
    avg_ocean berisi skor OCEAN skala Likert 1–5
    dikonversi ke persen (0–100%) untuk BAR CHART
    """

    bar_chart = []

    for trait in ["O", "C", "E", "A", "N"]:
        value = avg_ocean.get(trait, 1)

        # VALIDASI RANGE LIKERT
        if value < 1 or value > 5:
            raise ValueError(
                f"Invalid Likert value for {trait}: {value}. Expected 1–5"
            )

        percent = round(((value - 1) / 4) * 100, 1)

        bar_chart.append({
            "trait": trait,
            "label": {
                "O": "Openness",
                "C": "Conscientiousness",
                "E": "Extraversion",
                "A": "Agreeableness",
                "N": "Neuroticism"
            }[trait],
            "value": percent,              # UNTUK TINGGI BAR
            "raw_likert": round(value, 2), # UNTUK TOOLTIP
            "color": OCEAN_COLORS[trait]
        })

    return bar_chart

def aggregate_ocean_profile(results):
    if not results:
        return None

    traits = ["O", "C", "E", "A", "N"]
    total = {t: 0.0 for t in traits}
    count = 0

    for r in results:
        ocean = r.get("prediction_adjusted")
        if not ocean:
            continue

        for t in traits:
            total[t] += ocean.get(t, 0)
        count += 1

    if count == 0:
        return None

    avg = {t: round(total[t] / count, 3) for t in traits}
    dominant = max(avg, key=avg.get)

    conclusion, suggestion = generate_global_conclusion(avg, dominant)
    bar_chart = ocean_to_bar_chart(avg)


    return {
    "average_ocean_likert": avg,
    "average_ocean_percent": {
        t: round(((avg[t] - 1) / 4) * 100, 1) for t in traits
    },
    "dominant_trait": dominant,
    "bar_chart": bar_chart,
    "scale_info": {
        "model_scale": "Likert 1–5",
        "visualization": "Bar chart",
        "percentage_formula": "(value - 1) / 4 * 100"
    },
    "conclusion": conclusion,
    "suggestion": suggestion,
    "total_text_analyzed": count
    }
# =========================
# GENERATE PERSONA FUNCTION
# =========================
def generate_persona_profile(scores):
    best_label = "Seimbang"
    best_desc = "adaptif, fleksibel, dan tidak ekstrem pada satu trait"
    best_score = -float("inf")

    for label, cond, desc in PERSONA_RULES:
        if cond(scores):
            # Hitung "score dominasi" sebagai selisih trait tertinggi terhadap trait lain
            # Misal, N dominan → N - rata-rata trait lain
            dominant_trait = max(scores, key=scores.get)
            dominance = scores[dominant_trait] - sum(v for k,v in scores.items() if k != dominant_trait)/4
            if dominance > best_score:
                best_score = dominance
                best_label = label
                best_desc = desc

    return [f"Kepribadian : {best_label} — {best_desc}"]

def fetch_user_tweets(access_token: str, max_results: int = 10):
    # Gunakan OAuth2 User Context
    client = tweepy.Client(
        client_id=TWITTER_CLIENT_ID,
        client_secret=os.getenv("TWITTER_CLIENT_SECRET"),
        access_token=access_token,
        token_type="user"
    )

    me = client.get_me()
    user_id = me.data.id

    tweets = client.get_users_tweets(
        id=user_id,
        max_results=max_results,
        exclude=["retweets", "replies"]
    )

    if not tweets.data:
        return ""

    texts = [t.text for t in tweets.data]
    return " ".join(texts)
def generate_ocean_chart(ocean_scores: dict):
    traits = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ]

    values = [
        ocean_scores["O"],
        ocean_scores["C"],
        ocean_scores["E"],
        ocean_scores["A"],
        ocean_scores["N"]
    ]

    # Warna khusus OCEAN (premium & konsisten)
    colors = [
        "#3B82F6",  # Openness - Blue
        "#10B981",  # Conscientiousness - Green
        "#F59E0B",  # Extraversion - Orange
        "#8B5CF6",  # Agreeableness - Purple
        "#EF4444",  # Neuroticism - Red
    ]

    plt.figure(figsize=(6, 6))

    plt.pie(
        values,
        labels=traits,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10}
    )

    plt.title("OCEAN Personality Composition", fontsize=14, fontweight="bold")
    plt.axis("equal")  # Biar lingkaran sempurna

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150, transparent=True)
    plt.close()

    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return img_base64

def get_oauth_handler():
    return OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_REDIRECT_URI,
        scope=TWITTER_SCOPES,
    )
oauth = get_oauth_handler()
# ==========================
# ROUTES
# ==========================
@app.on_event("startup")
def startup_event():
    global ontology_df, SUBTRAITS, LEXICAL_SIZE, subtrait2id
    global LEXICON, ONT_EMBEDDINGS, ONT_META
    global tokenizer, model

    logger.info("🚀 Startup loading ontology & model")

    # ===============================
    # 1️⃣ LOAD ONTOLOGY (WAJIB DULU)
    # ===============================
    ontology_df = pd.read_csv(ONTOLOGY_CSV)

    if ontology_df is None or ontology_df.empty:
        raise RuntimeError("Ontology CSV kosong / gagal dibaca")

    ontology_df["tokens"] = ontology_df["lexeme"].astype(str).apply(lambda x: x.split("_"))

    if "strength" not in ontology_df.columns:
        ontology_df["strength"] = 1.0

    SUBTRAITS = sorted(ontology_df["sub_trait"].dropna().unique())
    LEXICAL_SIZE = len(SUBTRAITS)

    if LEXICAL_SIZE == 0:
        raise RuntimeError("LEXICAL_SIZE = 0, ontology bermasalah")

    subtrait2id = {s: i for i, s in enumerate(SUBTRAITS)}

    LEXICON = defaultdict(list)
    for _, row in ontology_df.iterrows():
        LEXICON[row["sub_trait"]].append({
            "tokens": set(row["tokens"]),
            "strength": float(row["strength"]),
            "lexeme": row["lexeme"]
        })

    # ===============================
    # 2️⃣ LOAD ONTOLOGY EMBEDDING
    # ===============================
    ont_emb = torch.load(ONTOLOGY_EMB, map_location="cpu")
    ONT_EMBEDDINGS = ont_emb["embeddings"].numpy()
    ONT_META = ont_emb["meta"]

    # ===============================
    # 3️⃣ LOAD TOKENIZER & ENCODER
    # ===============================
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    encoder = AutoModel.from_pretrained(HF_REPO)

    # ===============================
    # 4️⃣ BUILD MODEL (SETELAH LEXICAL_SIZE ADA)
    # ===============================
    model = OceanModel(encoder, LEXICAL_SIZE)

    # ===============================
    # 5️⃣ LOAD WEIGHT (.bin)
    # ===============================
    state_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="pytorch_model.bin"
    )

    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    logger.info(f"✅ Startup OK | LEXICAL_SIZE={LEXICAL_SIZE}")


@app.get("/")
def root():
    return {
    "service": "SAPA OCEAN API",
    "device": DEVICE,
    "subtraits": LEXICAL_SIZE,
    "status": "OK"
}

AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"

@app.get("/auth/twitter/login")
def twitter_login(request: Request):

    code_verifier = generate_token(64)

    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode("utf-8")

    oauth = OAuth2Session(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_REDIRECT_URI,
        scope=TWITTER_SCOPES
    )

    state = generate_token(32)

    authorization_url, _ = oauth.authorization_url(
        AUTH_URL,
        state=state,                     
        code_challenge=code_challenge,
        code_challenge_method="S256"
    )


    request.session["oauth_state"] = state
    request.session["code_verifier"] = code_verifier

    return RedirectResponse(authorization_url)
FRONTEND_URL = "http://localhost:3000"

@app.get("/auth/twitter/callback")
def twitter_callback(request: Request, code: str, state: str):

    if state != request.session.get("oauth_state"):
        raise HTTPException(400, "Invalid OAuth state")

    oauth = OAuth2Session(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_REDIRECT_URI,
        scope=TWITTER_SCOPES,
        state=state
    )

    token = oauth.fetch_token(
        TOKEN_URL,
        code=code,
        code_verifier=request.session["code_verifier"],
        client_secret=os.getenv("TWITTER_CLIENT_SECRET"),
    )

    request.session["twitter_access_token"] = token["access_token"]

    # 🔥 REDIRECT KE FRONTEND
    return RedirectResponse(
        url=f"{FRONTEND_URL}?twitter=success",
        status_code=302
    )
@app.get("/auth/twitter/me")
def twitter_me(request: Request):
    access_token = request.session.get("twitter_access_token")
    if not access_token:
        raise HTTPException(401, "Not authenticated")

    client = tweepy.Client(access_token)
    me = client.get_me()
    return {"username": me.data.username}

@app.get("/predict/twitter/check")
def twitter_check(request: Request):
    return {
        "logged_in": bool(
            request.session.get("twitter_access_token")
        )
    }
@app.post("/predict/twitter")
def predict_from_twitter(request: Request):

    access_token = request.session.get("twitter_access_token")
    if not access_token:
        raise HTTPException(401, "Twitter not authenticated")

    twitter_text = fetch_user_tweets(access_token)
    if not twitter_text.strip():
        raise HTTPException(404, "No tweets found")

    return run_ocean_pipeline(
        text=twitter_text,
        username=tweepy.Client(access_token).get_me().data.username
    )
import logging
logging.basicConfig(level=logging.INFO)
@app.post("/predict/twitter/profile")
def predict_other_profile(data: dict, request: Request):
    try:
        profile_url = data.get("profile_url")
        if not profile_url:
            raise HTTPException(400, "Missing profile_url")

        username = profile_url.rstrip("/").split("/")[-1].replace("@", "")
        logging.info(f"Fetching tweets for {username}")

        TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
        if not TWITTER_BEARER_TOKEN:
            raise HTTPException(500, "TWITTER_BEARER_TOKEN not set in .env")

        # App-only client untuk profile publik
        app_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

        user = app_client.get_user(username=username)
        logging.info(f"User found: {user.data}")

        tweets = app_client.get_users_tweets(
            id=user.data.id,
            max_results=10,
            exclude=["retweets","replies"]
        )

        if not tweets.data:
            raise HTTPException(404, "No tweets found")

        text = " ".join(t.text for t in tweets.data)
        return run_ocean_pipeline(text=text, username=username)

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in /predict/twitter/profile: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")
def run_ocean_pipeline(text: str, username: str | None = None):

    if tokenizer is None or model is None:
        raise HTTPException(503, "Model not ready")

    # ===== Lexical =====
    lexical, coverage, subtraits, evidence = build_lexical_vector_with_analysis(text)
    lexical = lexical.to(DEVICE)

    if lexical.dim() == 1:
        lexical = lexical.unsqueeze(0)

    # ===== Tokenize =====
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    # ===== Inference =====
    with torch.no_grad():
        out = model(
            enc["input_ids"],
            enc["attention_mask"],
            lexical
        )

    raw = {
        "O": round(out[0, 0].item(), 3),
        "C": round(out[0, 1].item(), 3),
        "E": round(out[0, 2].item(), 3),
        "A": round(out[0, 3].item(), 3),
        "N": round(out[0, 4].item(), 3),
    }

    dominant, adjusted = adjust_ocean_by_keywords(raw, text)
    adjusted = apply_emotional_keyword_adjustment(text, adjusted)
    dominant = max(adjusted, key=adjusted.get)

    explanation, suggestion = generate_explanation_suggestion_super(
        text, adjusted, evidence
    )

    try:
        chart = generate_ocean_chart(adjusted)
    except Exception:
        chart = None

    return {
        "username": username,
        "highlighted_text": highlight_keywords_in_text(text, evidence),
        "prediction_adjusted": adjusted,
        "dominant_trait": dominant,
        "personality_profile": generate_persona_profile(adjusted),
        "explanation": explanation,
        "suggestion": suggestion,
        "ocean_chart_base64": chart
    }

@app.post("/predict/excel")
async def predict_from_excel(file: UploadFile = File(...)):
    """
    Upload file Excel (.xlsx) dengan kolom 'text' berisi teks.
    Endpoint akan memproses tiap teks dan mengembalikan prediksi OCEAN.
    """
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "File harus berformat .xlsx")

    try:
        # Baca Excel ke DataFrame
        df = pd.read_excel(file.file)

        if "text" not in df.columns:
            raise HTTPException(400, "Excel harus memiliki kolom 'text'")

        results = []

        # Loop tiap baris teks
        for idx, row in df.iterrows():
            text = str(row["text"])
            if not text.strip():
                continue

            result = run_ocean_pipeline(text=text, username=None)
            result["row_index"] = idx
            results.append(result)

        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(500, f"Error memproses file Excel: {str(e)}")
@app.post("/predict/excel/profile")
async def predict_from_excel_profile(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "File harus berformat .xlsx")

    try:
        df = pd.read_excel(file.file)

        if "text" not in df.columns:
            raise HTTPException(400, "Excel harus memiliki kolom 'text'")

        results = []

        for idx, row in df.iterrows():
            text = str(row["text"])
            if not text.strip():
                continue

            result = run_ocean_pipeline(text=text, username=None)
            result["row_index"] = idx
            results.append(result)

        profile_summary = aggregate_ocean_profile(results)

        return {
            "status": "success",
            "row_results": results,
            "profile_summary": profile_summary
        }

    except Exception as e:
        raise HTTPException(500, f"Error memproses file Excel: {str(e)}")

@app.post("/predict")
def predict(data: TextInput):
    lexical, coverage, subtraits, evidence = build_lexical_vector_with_analysis(data.text)
    lexical = lexical.to(DEVICE)

    enc = tokenizer(
        data.text,
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
        "O": round(out[0,0].item(), 3),
        "C": round(out[0,1].item(), 3),
        "E": round(out[0,2].item(), 3),
        "A": round(out[0,3].item(), 3),
        "N": round(out[0,4].item(), 3),
    }

    # 1️⃣ Adjustment awal (aturan sederhana)
    dominant, adjusted = adjust_ocean_by_keywords(raw, data.text)

    # 2️⃣ Adjustment emosional & sosial berbobot (INI YANG KAMU TAMBAHKAN)
    adjusted = apply_emotional_keyword_adjustment(
        data.text,
        adjusted
    )

    # 3️⃣ Tentukan ulang dominant trait
    dominant = max(adjusted, key=adjusted.get)

    # 4️⃣ Baru buat explanation & suggestion
    explanation, suggestion = generate_explanation_suggestion_super(
        data.text,
        adjusted,
        evidence
    )

    return {
        "input_text": data.text,
        "highlighted_text": highlight_keywords_in_text(data.text, evidence),
        "prediction_raw": raw,
        "prediction_adjusted": adjusted,
        "dominant_trait": dominant,
        "personality_profile": generate_persona_profile(adjusted),
        "ontology_analysis": {
            "coverage_percent": coverage,
            "active_subtraits": subtraits
        },
        "lexical_evidence": evidence,
        "ontology_expansion_candidates": expand_ontology_candidates(data.text),
        "explanation": explanation,
        "suggestion": suggestion
    }
