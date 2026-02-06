import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import tweepy
import math
import os
import base64
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
from fastapi.responses import StreamingResponse
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
# OCEAN ADJUSTMENT (DATA DRIVEN)
# ==========================
from collections import Counter
import re
def adjust_ocean_by_keywords(scores: dict, text: str):
    adjusted = scores.copy()
    text_lower = text.lower()
    tokens = re.findall(r'\w+', text_lower)
    counter = Counter(tokens)
    # Extreme score
    adjusted["EXTREME_ALERT"] = 0
    # ==========================
    # 1. WORD BASED
    # ==========================
    for trait, keywords in TRAIT_KEYWORDS.items():
        if trait not in KEYWORD_TRAIT_MAP:
            continue
        weights = KEYWORD_TRAIT_MAP[trait]
        for word in keywords:
            if word in counter:
                freq = counter[word]
                # extreme detect
                if trait == "EXTREME_NEGATIVE":
                    adjusted["EXTREME_ALERT"] += freq * 2

                for ocean_dim, w in weights.items():
                    adjusted[ocean_dim] += w * freq
    # ==========================
    # 2. PHRASE BASED
    # ==========================
    for trait, keywords in TRAIT_KEYWORDS.items():
        if trait not in KEYWORD_TRAIT_MAP:
            continue
        weights = KEYWORD_TRAIT_MAP[trait]
        for phrase in keywords:

            if " " in phrase and phrase in text_lower:

                if trait == "EXTREME_NEGATIVE":
                    adjusted["EXTREME_ALERT"] += 3

                for ocean_dim, w in weights.items():
                    adjusted[ocean_dim] += w * 1.5
    # ==========================
    # 3. NORMALIZATION
    # ==========================
    for k in ["O", "C", "E", "A", "N"]:
        adjusted[k] = max(1.0, min(5.0, adjusted[k]))
    # only OCEAN for dominant
    ocean_only = {k:v for k,v in adjusted.items() if k in ["O","C","E","A","N"]}
    dominant = max(ocean_only, key=ocean_only.get)
    return dominant, adjusted
# ==========================
# KEYWORD → OCEAN MAPPING
# ==========================
KEYWORD_TRAIT_MAP = {
# ===== EMOSI NEGATIF =====
"ANGER_EMO": {"N": 0.5, "A": -0.2},
"SAD_EMO": {"N": 0.4, "O": 0.1},
"ANXIETY_EMO": {"N": 0.6, "E": -0.2},
"MENTAL_UNSTABLE_N": {"N": 1.0},
"NEGATIVE_SOCIAL": {"N": 0.4, "A": -0.3, "E": -0.2},
# ===== SOSIAL =====
"POSITIVE_SOCIAL": {"E": 0.4, "A": 0.4, "N": -0.2},
"EXTRAVERSION_E": {"E": 0.6},
"E_SOCIAL_DEPENDENCY": {"E": 0.4, "A": 0.2},
"COLLABORATION": {"A": 0.6, "E": 0.3, "C": 0.2},
# ===== RELASIONAL =====
"RELATIONSHIP_AFFECTION": {"A": 0.7, "E": 0.2},
"EMPATHY_HARMONY_A": {"A": 0.8, "N": -0.3},
"TRUST": {"A": 0.5},
# ===== KOGNITIF =====
"CREATIVE_DISCUSSION_A": {"O": 0.6, "E": 0.2},
"INTROSPECTION": {"O": 0.5, "N": 0.1},
# ===== PRODUKTIVITAS =====
"DISCIPLINE_C": {"C": 0.9, "N": -0.2},
"ACHIEVEMENT": {"C": 0.6, "E": 0.2},
# ===== EMOSI POSITIF =====
"EMO_POSITIVE": {"A": 0.3, "E": 0.3, "N": -0.2},
"MENTAL_UNSTABLE_N": {"N": 1.0},
"EXTREME_NEGATIVE": {
    "N": 2.5,
    "E": -0.8,
    "A": -0.6,
    "C": -0.6
},
}
# ==========================
# EMOTIONAL ADJUSTMENT (FINE TUNING)
# ==========================
def apply_emotional_keyword_adjustment(text: str, scores: dict, o_reduce: float = 0.2):
    adjusted = scores.copy()
    text_lower = text.lower()
    tokens = re.findall(r'\w+', text_lower)
    counter = Counter(tokens)
    # ==========================
    # 1. EMOTIONAL BALANCE SCORE
    # ==========================
    neg_score = 0
    pos_score = 0
    for word in tokens:
        if word in ANGER_EMO or word in SAD_EMO or word in ANXIETY_EMO:
            neg_score += counter[word]
        if word in EMO_POSITIVE or word in POSITIVE_SOCIAL or word in TRUST:
            pos_score += counter[word]
    # ==========================
    # 2. ADJUST BASED ON BALANCE
    # ==========================
    if neg_score > pos_score:
        diff = min(2.0, (neg_score - pos_score) * 0.15)
        adjusted["N"] += diff
        adjusted["E"] -= diff * 0.4
    elif pos_score > neg_score:
        diff = min(2.0, (pos_score - neg_score) * 0.15)
        adjusted["A"] += diff
        adjusted["E"] += diff * 0.3
        adjusted["N"] -= diff * 0.3
    # ==========================
    # 3. O REDUCTION (CONTEXT AWARE)
    # ==========================
    creative_hits = 0

    for w in EMO_POSITIVE + CREATIVE_DISCUSSION_A + INTROSPECTION:
        if w in counter:
            creative_hits += counter[w]
    # Turunkan O hanya kalau TIDAK kreatif
    if creative_hits == 0:
        adjusted["O"] -= o_reduce
    # ==========================
    # 4. NORMALIZE
    # ==========================
    for k in ["O", "C", "E", "A", "N"]:
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
    scores = adjusted.copy()
    scores["E"] += len(evidence.get("POSITIVE_SOCIAL", [])) * 0.3
    scores["A"] += len(evidence.get("EMPATHY_HARMONY_A", [])) * 0.4
    scores["C"] += len(evidence.get("DISCIPLINE_C", [])) * 0.4
    scores["O"] += len(evidence.get("CREATIVE_DISCUSSION_A", [])) * 0.4
    scores["N"] += len(evidence.get("ANXIETY_EMO", [])) * 0.5
    return max(scores, key=scores.get)
def normalize_scores(scores, min_val=1.0, max_val=5.0):

    for k in scores:
        scores[k] = max(min(scores[k], max_val), min_val)

    return scores
PERSONA_RULES = [
    # ==========================
    # ⚠️ EXTREME / HIGH RISK
    # ==========================
    ("Krisis Emosional Tinggi",
     lambda s: s.get("EXTREME_ALERT", 0) >= 3 or s["N"] >= 4.7,
     "menunjukkan tanda tekanan psikologis berat, putus asa, atau risiko menyakiti diri"),
    ("Depresi Mendalam",
     lambda s: s.get("EXTREME_ALERT", 0) >= 2 and s["E"] <= 2.5 and s["N"] >= 4.2,
     "menarik diri, kehilangan motivasi, dan mengalami kesedihan intens"),
    ("Burnout Mental Berat",
     lambda s: s.get("EXTREME_ALERT", 0) >= 2 and s["C"] <= 2.8 and s["N"] >= 4.0,
     "kelelahan emosional ekstrem, kehilangan arah, dan kehabisan energi"),
    # ==========================
    # EMOSI NEGATIF
    # ==========================
    # CEMAS → REFLEKTIF, BANYAK PIKIR
    ("Cemas & Overthinking",
    lambda s: s["N"] >= 3.6 and s["O"] >= 3.4 and s["C"] >= 3.0,
    "mudah khawatir, banyak berpikir, reflektif terhadap masalah"),
    # TEMPRAMENTAL → REAKTIF, KURANG KONTROL
    ("Tempramental",
    lambda s: s["N"] >= 4.0 and s["A"] <= 2.9 and s["C"] <= 3.0,
    "emosional, mudah tersulut, impulsif saat tertekan"),
    ("Melankolis Reflektif",
     lambda s: s["N"] >= 3.3 and s["O"] >= 3.3 and s["E"] <= 3.0,
     "sering merenung, introspektif, sensitif terhadap perasaan"),
    # ==========================
    # STABILITAS EMOSI
    # ==========================
    ("Stabil Emosional",
     lambda s: s["N"] <= 2.5,
     "tenang, terkendali, mampu mengelola tekanan dengan baik"),
    ("Tangguh Mental",
     lambda s: s["C"] >= 3.5 and s["N"] <= 3.0,
     "kuat secara mental, tidak mudah menyerah, fokus solusi"),
    # ==========================
    # SOSIAL & EKSTROVERSI
    # ==========================
    ("Ekstrovert Sosial",
     lambda s: s["E"] >= 3.8 and s["A"] >= 3.2,
     "percaya diri, aktif berinteraksi, mudah bergaul"),

    ("Sosial Ekspresif",
     lambda s: s["E"] >= 3.5 and s["O"] >= 3.2,
     "komunikatif, ekspresif, suka berbagi ide"),

    ("Introvert Mandiri",
     lambda s: s["E"] <= 2.8 and s["C"] >= 3.2,
     "mandiri, fokus, nyaman bekerja sendiri"),
    # ==========================
    # RELASIONAL & CINTA
    # ==========================
    ("Romantis Afektif",
     lambda s: s["A"] >= 3.6 and s["A"] >= s["O"] + 0.2,
     "hangat, penuh perhatian, berorientasi pada hubungan"),

    ("Empatik Caregiver",
     lambda s: s["A"] >= 3.7 and s["N"] <= 3.2,
     "peduli, protektif, senang membantu orang lain"),

    ("Relasional Selektif",
     lambda s: s["A"] >= 3.3 and s["E"] <= 3.0,
     "ramah namun berhati-hati dalam memilih relasi"),
    # ==========================
    # KREATIVITAS & VISI
    # ==========================
    ("Visioner Kreatif",
     lambda s: s["O"] >= 3.7 and s["O"] >= s["A"] + 0.2,
     "imajinatif, visioner, berpikir jauh ke depan"),

    ("Pemikir Inovatif",
     lambda s: s["O"] >= 3.5 and s["E"] >= 3.2,
     "aktif menciptakan ide baru dan solusi kreatif"),

    ("Reflektif Analitis",
     lambda s: s["O"] >= 3.3 and s["C"] >= 3.3,
     "mendalam, sistematis, kritis dalam berpikir"),
    # ==========================
    # DISIPLIN & KERJA
    # ==========================
    ("Perfeksionis Produktif",
     lambda s: s["C"] >= 3.8 and s["C"] >= s["N"] + 0.2,
     "teliti, terstruktur, menuntut standar tinggi"),

    ("Ambisius Visioner",
     lambda s: s["C"] >= 3.5 and s["O"] >= 3.5,
     "berorientasi prestasi, berpikir strategis"),

    ("Pragmatis Efisien",
     lambda s: s["C"] >= 3.3 and s["E"] >= 3.2,
     "praktis, fokus hasil, cepat mengambil keputusan"),

    ("Gigih & Persisten",
     lambda s: s["C"] >= 3.4 and s["N"] <= 3.0,
     "konsisten, tahan tekanan, tidak mudah menyerah"),
    # ==========================
    # KESEIMBANGAN
    # ==========================
    ("Seimbang Adaptif",
     lambda s: all(2.8 <= s[k] <= 3.5 for k in ["O","C","E","A","N"]),
     "fleksibel, stabil, mampu menyesuaikan diri di berbagai situasi"),
]

# ==========================
# GLOBAL CONCLUSION
# ==========================
def generate_global_conclusion(avg, dominant):
    O, C, E, A, N = avg["O"], avg["C"], avg["E"], avg["A"], avg["N"]

    conclusion = (
        f"Secara keseluruhan, trait kepribadian paling dominan adalah {dominant}. Individu ini cenderung "
    )
    # PRIORITY EXTREME
    if avg.get("EXTREME_ALERT", 0) >= 3:
        conclusion = (
            "Terdapat indikasi tekanan emosional yang sangat tinggi dan risiko kesehatan mental."
        )

        suggestion = (
            "Sangat disarankan untuk segera mencari dukungan profesional, "
            "berbicara dengan orang terpercaya, atau menghubungi layanan bantuan psikologis."
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
    # Insight tambahan dari Neuroticism
    if N <= 2.5:
        conclusion += " Tingkat kestabilan emosi tergolong baik."
    elif N >= 3.6:
        conclusion += " Terdapat kecenderungan emosi negatif yang cukup tinggi."

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
def likert_to_percent(value, scale=5):
    if value is None:
        return 0
    return round((float(value) / scale) * 100, 2)
def build_excel_rows(results):
    rows = []

    for r in results:
        scores = r["prediction_adjusted"]

        rows.append({
            "text": re.sub(r"<.*?>", "", r.get("highlighted_text", "")),
            "O (%)": likert_to_percent(scores["O"]),
            "C (%)": likert_to_percent(scores["C"]),
            "E (%)": likert_to_percent(scores["E"]),
            "A (%)": likert_to_percent(scores["A"]),
            "N (%)": likert_to_percent(scores["N"]),
            "kepribadian": ", ".join(r.get("personality_profile", [])),
            "solusi": r.get("suggestion", "")
        })

    return pd.DataFrame(rows)

def dataframe_to_excel_bytes(df_detail, profile_summary=None):
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_detail.to_excel(writer, index=False, sheet_name="Detail")

        if profile_summary is not None:
            cols = len(df_detail.columns)
            start_row = len(df_detail) + 2

            def pad(values):
                return values + [""] * (cols - len(values))

            summary_rows = [
                pad(["SUMMARY"]),
                pad(["Average O (%)", likert_to_percent(profile_summary["average_ocean_likert"]["O"])]),
                pad(["Average C (%)", likert_to_percent(profile_summary["average_ocean_likert"]["C"])]),
                pad(["Average E (%)", likert_to_percent(profile_summary["average_ocean_likert"]["E"])]),
                pad(["Average A (%)", likert_to_percent(profile_summary["average_ocean_likert"]["A"])]),
                pad(["Average N (%)", likert_to_percent(profile_summary["average_ocean_likert"]["N"])]),
                pad(["Dominant Trait", profile_summary["dominant_trait"]]),
                pad(["Conclusion", profile_summary["conclusion"]]),
                pad(["Suggestion", profile_summary["suggestion"]]),
                pad(["Total Text", profile_summary["total_text_analyzed"]]),
            ]


            df_summary = pd.DataFrame(summary_rows, columns=df_detail.columns)
            df_summary.to_excel(
                writer,
                index=False,
                header=False,
                sheet_name="Detail",
                startrow=start_row
            )

    buffer.seek(0)
    return buffer

def excel_buffer_to_base64(buffer: BytesIO):
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
@app.post("/predict/excel")
async def predict_from_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "File harus berformat .xlsx")

    df = pd.read_excel(file.file)

    if "text" not in df.columns:
        raise HTTPException(400, "Excel harus memiliki kolom 'text'")

    results = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        if not text.strip():
            continue

        r = run_ocean_pipeline(text=text)
        r["row_index"] = idx
        results.append(r)

    # === BUILD EXCEL ===
    df_detail = build_excel_rows(results)
    excel_buffer = dataframe_to_excel_bytes(df_detail)
    excel_b64 = excel_buffer_to_base64(excel_buffer)

    return {
        "status": "success",
        "total_rows": len(results),
        "results": results,
        "excel": {
            "filename": "ocean_result.xlsx",
            "content_base64": excel_b64
        }
    }
@app.post("/predict/excel/profile")
async def predict_from_excel_profile(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "File harus berformat .xlsx")

    df = pd.read_excel(file.file)

    if "text" not in df.columns:
        raise HTTPException(400, "Excel harus memiliki kolom 'text'")

    results = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        if not text.strip():
            continue

        r = run_ocean_pipeline(text=text)
        r["row_index"] = idx
        results.append(r)

    # ===== PROFILE AGGREGATION =====
    profile_summary = aggregate_ocean_profile(results)

    # ===== EXCEL =====
    df_detail = build_excel_rows(results)
    excel_buffer = dataframe_to_excel_bytes(
        df_detail,
        profile_summary=profile_summary
    )
    excel_b64 = excel_buffer_to_base64(excel_buffer)

    return {
        "status": "success",
        "total_text": len(results),
        "row_results": results,
        "profile_summary": profile_summary,
        "excel": {
            "filename": "ocean_profile_result.xlsx",
            "content_base64": excel_b64
        }
    }

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
