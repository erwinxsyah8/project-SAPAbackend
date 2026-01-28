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
# =========================
# EMOSI NEGATIF / KESAL, SEDIH, CEMAS
# =========================
EMO_NEGATIVE = [
    "sedih","kesedihan","kecewa","kekecewaan","frustasi","putus asa","kesal","kemarahan",
    "marah","gelisah","cemas","kekhawatiran","menangis","stress","stresnya","terpuruk",
    "bosan","capek","kelelahan","tertekan","frustrasi","murung","gundah","kehilangan",
    "tak berdaya","jengkel","panik","khawatir berlebihan","curiga","resah","was-was",
    "overthinking","terluka","terluka hati","kebingungan","sepi","minder","terasing",
    "canggung","grogi","ragu","keraguan","malu","malunya","terasingkan","pemalu","anti-sosial",
    "bingung","gelisah hati","deg-degan","frustasi sosial","putus harapan","galau","merana",
    "hampa","pilu","sayu","sengsara","duka","luka hati","meresah","pilu hati","melankolis",
    "sendu","tersiksa","kecewanya","menyesal","menangisi","meratap","terluka",
    "frustrasi berat","kehilangan harapan","gelisah mental","emosional","putus asa berat",
    "murung mendalam","stress berat","panik mendadak","curiga berlebihan",
    # 2 kata
    "tidak bahagia","sangat sedih","kecewa berat","sangat frustasi","cemas berlebihan",
    "murung mendalam","khawatir berlebihan","gelisah mental","overthinking berlebihan",
    "putus asa","tertekan berat","kehilangan besar","sedih mendalam","gelisah parah",
    "merasa sedih","murung berat","frustasi sosial","emosional berat","stress berat",

    # 3 kata
    "putus asa berat","kehilangan harapan besar","gelisah hati mendalam","terluka hati berat",
    "merasa sangat sedih","khawatir dan cemas","murung dan tertekan","frustasi sangat berat",
    "sangat sedih dan kecewa","cemas berlebihan dan khawatir","kecewa dan putus asa"
]

# Variants imbuhan
EMO_NEGATIVE += [f"me{w}" for w in ["marah","gelisah","frustasi","cemas","jengkel","murung"]]
EMO_NEGATIVE += [f"ter{w}" for w in ["tekan","murung","sedih","cemas","curiga","galau","frustrasi"]]
EMO_NEGATIVE += [f"ke{w}" for w in ["cewa","sedih","luka","kehilangan","kekecewaan"]]

# ====== ANGER / MARAH / FRUSTRASI
ANGER_EMO = [
    "marah","kesal","jengkel","geram","murka","emosi","frustrasi","panik","mendidih","berang",
    "berbenci","bete","jengkel hati","marah-marah","termarah","memarah","geramnya","mendongkol",
    "menggeram","geram hati","amarah","emosinya","keselnya","resah","kebencian",
    "membanting","menyerang","mendam","mengeram","mencak-mencak","frustrasi berat","marah mendadak",
    # 2 kata
    "sangat marah","amarah besar","frustrasi berat","marah mendadak","geram luar biasa",
    "kesal berat","emosi negatif","jengkel berat","marah dan frustrasi",

    # 3 kata
    "marah tidak terkendali","frustrasi sangat berat","kesal luar biasa","jengkel dan marah",
    "emosi sangat tinggi","marah dan kesal"
]
ANGER_EMO += [f"me{w}" for w in ["marah","jengkel","geram","murka","benci"]]
ANGER_EMO += [f"ter{w}" for w in ["marah","jengkel","geram","murka"]]

# ====== SADNESS / SEDIH
SAD_EMO = [
    "sedih","kesedihan","kecewa","kekecewaan","putus asa","menangis","murung","terpuruk",
    "merana","galau","hampa","pilu","sayu","sengsara","duka","luka hati","meresah",
    "kehilangan","pilu hati","melankolis","sendu","tersiksa","kecewanya","menyesal",
    "menangisi","meratap","terluka","terluka hati","frustasi emosional","murung mendalam",
    "putus asa berat","kehilangan harapan",
     # 2 kata
    "sangat sedih","kecewa berat","murung mendalam","putus asa berat","gelisah hati",
    "sedih parah","merasa sedih","hati sedih","kehilangan besar","murung dan sedih",

    # 3 kata
    "merasa kehilangan besar","sedih dan frustasi","murung dan tertekan","sedih dan kecewa",
    "putus asa sangat berat","gelisah dan cemas"
]
SAD_EMO += [f"ke{w}" for w in ["cewa","sedih","luka","kehilangan","kekecewaan"]]
SAD_EMO += [f"ter{w}" for w in ["sedih","kecewa","murung","galau"]]

# ====== ANXIETY / CEMAS / TAKUT
ANXIETY_EMO = [
    "cemas","gelisah","khawatir","kekhawatiran","was-was","overthinking","bingung",
    "takut","takutnya","panik","resah","gugup","grogi","ragu","keraguan","tertekan",
    "tekanan","deg-degan","gelisah hati","khawatir berlebihan","curiga","waswas","cemasnya",
    "gelisahnya","cemas berlebihan","tergugup","tertekan","tercemas","tercuriga",
    "grogi berlebihan","kekhawatiran mendalam","gelisah mental","takut berat",
    # 2 kata
    "cemas berlebihan","khawatir berlebihan","gelisah hati","takut berat","deg-degan parah",
    "cemas parah","khawatir tinggi","gelisah berlebihan","was-was berat",

    # 3 kata
    "cemas dan gelisah","khawatir terus menerus","takut sangat berat","gelisah dan khawatir",
    "deg-degan sangat parah","cemas dan takut"
]
ANXIETY_EMO += [f"ter{w}" for w in ["tekan","gugup","cemas","curiga","khawatir"]]
ANXIETY_EMO += [f"me{w}" for w in ["cemas","gelisah","khawatir"]]

# ====== EMOSI POSITIF / SENANG / BAHAGIA
EMO_POSITIVE = [
    "senang","kesenangan","bahagia","kebahagiaan","puas","kepuasan","bangga","kebanggaan",
    "gembira","ceria","lega","ketenangan","termotivasi","motivasinya","tenang","optimis",
    "antusias","relaks","positif","semangat","senyum","puas hati","riang","bersemangat",
    "terinspirasi","syukur","damai","berenergi","senyum-senyum","bermotif positif",
    "senangnya","bahagianya","lega hati","bahagia sekali","puas banget","termotivasi","bersemangat",
    "riang gembira","senyum lebar","puas luar biasa","optimis tinggi","bahagia mendalam","antusiasme tinggi",
    # 2 kata
    "sangat senang","bahagia sekali","puas hati","riang gembira","senyum lebar",
    "senang dan puas","termotivasi tinggi","tenang dan damai","optimis tinggi","riang dan bahagia",

    # 3 kata
    "sangat bahagia sekali","riang dan gembira","puas dan senang","senang dan bersemangat",
    "bahagia dan tenang","termotivasi dan antusias"
]

# ====== SOSIAL / TRUST / RELATIONSHIP
NEGATIVE_SOCIAL = [
    "takut","takutnya","ketakutan","cemas","kecemasan","tidak percaya diri","percaya diri rendah",
    "menyendiri","sendiri","menjauhi","malu","malunya","grogi","ragu","keraguan","tertekan",
    "tekanan","khawatir","kekhawatiran","menjauh","isolasi","mengasingkan","bingung","kebingungan",
    "terasing","canggung","gelisah","was-was","sepi","minder","terasingkan","curiga","pemalu",
    "anti-sosial","resah","menghindar","overthinking","terluka","terluka hati","frustrasi sosial",
    "terasing dari kelompok","menjauh dari teman","isolasi sosial","tidak nyaman berinteraksi",
    # 2 kata
    "tidak suka","tidak senang","benci pada","tidak nyaman","menjauhi orang",
    "tidak percaya","menghindar dari","tidak mau berinteraksi","tidak ingin berinteraksi",
    "tidak peduli","tidak menghargai","mengabaikan orang",

    # 3 kata
    "tidak suka orang","tidak nyaman berinteraksi","menjauhi orang lain","tidak percaya diri",
    "tidak peduli orang","mengabaikan orang lain","tidak ingin berinteraksi"
]

POSITIVE_SOCIAL = [
    "bertemu","bertemunya","ngobrol","ngobrolin","berbagi","membantu","hangout","bersosialisasi",
    "berinteraksi","teman","teman-teman","komunikasi","diskusi","kerjasama","bergaul","acara",
    "saling","kerabat","mendekat","bersenda","tertawa bersama","berkenalan","jaringan","teamwork",
    "ramah","ceria","humoris","aktif","berpartisipasi","sosial","berteman","berkumpul",
    "kolaborasi","kerjasama tim","mendukung","memotivasi","bekerjasama","menghargai teman",
    
]
EXTRAVERSION_E = [
    "aktif",
  "diskusi",
  "ngobrol",
  "ramai",
  "keramaian",
  "sosial",
  "interaksi",
  "bergaul",
  "berkumpul",
  "kerja tim",
  "kolaboratif",
  "presentasi",
  "depan umum",
  "kenal orang baru",
  "akrab",
  "nimbrung",
  "komunikasi",
  "berbincang",
  "bercerita",
  "berbagi cerita",
  "suasana hidup",
  "pusat perhatian",
  "banyak orang",
    "aktif berdiskusi","aktif berpartisipasi","aktif dalam diskusi","aktif diskusi kelompok",
    "aktif dalam kegiatan bersama","aktif dalam kegiatan kelompok","aktif dalam kegiatan sosial",
    "aktif bersosialisasi","aktif dalam acara sosial","aktif dalam lingkungan sosial",
    "aktif dalam percakapan kelompok","aktif ikut diskusi","aktif menyapa","aktif menyapa orang",
    "aktif saat berkumpul","aktif saat kerja tim","aktif saat ada acara",
    "energi meningkat saat berkumpul","energi naik saat bersama orang lain",
    "lebih hidup kalau ada teman","lebih hidup saat bersama orang lain",
    "lebih aktif kalau ada teman","lebih aktif saat ada teman",

    "mudah bergaul","mudah berinteraksi","mudah membangun komunikasi",
    "mudah menjalin komunikasi","mudah nyambung kalau diajak ngobrol",
    "mudah beradaptasi secara sosial","mudah bergaul di lingkungan baru",

    "senang berinteraksi","senang berinteraksi langsung","senang berinteraksi dengan orang baru",
    "senang berinteraksi dengan banyak orang","senang ngobrol","senang ngobrol lama",
    "senang ngobrol panjang lebar","senang ngobrol rame","senang ngobrol santai",
    "senang berbagi cerita","senang berbagi cerita ke orang lain",
    "senang berbagi pengalaman","senang bertemu banyak orang",
    "senang ikut kegiatan sosial","senang terlibat aktivitas sosial",
    "senang terlibat diskusi","senang diskusi santai",

    "suka ngobrol","suka ngobrol dengan banyak orang","suka ngobrol santai",
    "suka ngobrol sambil bercanda","suka ngobrol di berbagai situasi",
    "suka berbagi cerita","suka berbagi pengalaman",
    "suka aktivitas sosial","suka bergaul","suka interaksi langsung",
    "suka ikut acara kumpul","suka ikut acara rame",

    "lebih nyaman kerja tim","lebih nyaman kerja bareng tim",
    "lebih suka kerja tim","lebih suka diskusi langsung",
    "lebih suka kerja sambil ngobrol","lebih senang ngobrol daripada diam",

    "paling semangat kalau diskusi","paling semangat kalau ngobrol rame",
    "ngerasa lebih hidup saat ngobrol","ngerasa semangat kalau ngobrol lama",
    "kalau ada acara kumpul pasti ikut","kalau ada acara rame pasti ikut"
]
TRUST = [
    "peduli","menolong","percaya","percaya diri","loyal","setia","mendukung","mempercayai",
    "terbuka","saling percaya","menghargai","mengandalkan","solid","ramah","toleran",
    "mengerti","memaafkan","kooperatif","sopan","baik hati","humane","bekerjasama",
    "percaya penuh","percaya satu sama lain","percaya tim",
    "integritas","bertanggung jawab","mengayomi","memimpin","membimbing"
]

RELATIONSHIP_AFFECTION = [
    "sayang","cinta","kasih","peduli","rindu","kamu","kita","hubungan","bersama","pasangan",
    "pacar","kekasih","teman dekat","teman sejati","kekasih hati","hubungan romantis",
    "affeksi","kehangatan","kedekatan","kasih sayang","memelihara hubungan","intim"
]
# ====== IDE TERBUKA / DISKUSI MENDALAM / KREATIF SOSIAL (AGREEABLENESS DOMINANT)
CREATIVE_DISCUSSION_A = [
    "ide segar","pikiran terbuka","berpikir terbuka","diskusi ide","diskusi mendalam",
    "diskusi yang mendalam","diskusi lintas bidang","berbagi ide","bertukar ide",
    "kolaborasi ide","pemikiran terbuka","open minded","kreatif","kreativitas",
    "inovasi","solusi kreatif","pemikiran inovatif","ide kreatif",
    "menyukai diskusi","menikmati diskusi","diskusi intelektual",
    "mencari solusi bersama","pemecahan masalah bersama",
    "ide membantu","kreativitas membantu","berpikir terbuka membantu",
    "inovasi solusi","pemikiran lintas disiplin"
]
DISCIPLINE_C = [
    "disiplin",
  "tepat waktu",
  "tanggung jawab",
  "rapi",
  "teratur",
  "terstruktur",
  "sistematis",
  "konsisten",
  "teliti",
  "fokus",
  "kualitas",
  "deadline",
  "perencanaan",
  "rencana",
  "to-do list",
  "efisien",
  "produktif",
  "kerapihan",
  "komitmen",
  "target",
  "hasil maksimal",
  "tidak suka menunda",
  "tidak suka asal-asalan",
  "pekerjaan selesai",
  "sesuai rencana",
    "disiplin","bertanggung jawab","tanggung jawab","fokus","target","deadline",
    "tepat waktu","ketepatan waktu","konsisten","konsistensi","rapi","teratur",
    "terstruktur","terorganisir","sistematis","perencanaan","rencana","jadwal",
    "to-do list","daftar tugas","prioritas","efisien","produktif","teliti",
    "ketelitian","kualitas","hasil maksimal","hasil optimal","komitmen",
    "anti menunda","tidak suka menunda","tidak suka menunda pekerjaan",
    "menyelesaikan","menyelesaikan tugas","menyelesaikan pekerjaan",
    "selesai tepat waktu","sesuai rencana","standar kerja","kerja serius",
    "pekerjaan rapi","kerja rapi","kerja terencana","kerja sistematis",
    "kerja terstruktur","kerja terorganisir","fokus bekerja",
    "cek ulang","mengecek ulang","detail","setiap detail",
    "pekerjaan tuntas","sampai selesai","tidak setengah-setengah",
    "tidak asal-asalan","tidak terburu-buru","kualitas lebih penting",
    "menjaga kualitas","menjaga konsistensi","menjaga komitmen"
]
# ====== MENTAL UNSTABLE / OVERTHINKING / ANXIETY CORE (NEUROTICISM)
MENTAL_UNSTABLE_N = [
    "gugup","mudah gugup","mudah merasa gugup",
    "tidak tenang","merasa tidak tenang","sering tidak tenang",
    "perasaan tidak stabil","emosi tidak stabil",
    "pikiran kacau","pikiran mudah kacau","pikiran sering kacau",
    "pikiran negatif","pikiran negatif sering muncul",
    "pikiran negatif muncul berulang kali",
    "pikiran negatif muncul terus-menerus",
    "pikiran negatif sulit dikendalikan",
    "pikiran sering negatif","pikiran sering mengganggu",
    "pikiran tidak terkendali","pikiran terasa berat",
    "pikiran tidak pernah tenang",
    "stres","mudah stres","mudah merasa stres",
    "tidak aman","merasa tidak aman","sering merasa tidak aman",
    "cemas","kecemasan","gelisah","resah",
    "sulit rileks","sulit merasa rileks",
    "sulit tenang","sulit merasa tenang",
    "sulit merasa damai","sulit merasa nyaman",
    "sulit menenangkan diri","sulit menenangkan pikiran",
    "sulit merasa aman","sulit tenang sepenuhnya",
    "tidak nyaman tanpa sebab","gelisah tanpa sebab"
]

E_SOCIAL_DEPENDENCY = [
  "lebih suka kerja bareng",
  "lebih suka kerja kelompok",
  "diskusi ramai",
  "kurang semangat kalau sepi",
  "gelisah kalau sendirian",
  "lebih aktif kalau ada teman",
  "lebih nyaman kerja bareng"
]
EMPATHY_HARMONY_A = [
    "empati","berempati","simpati","iba","peduli",
    "tidak suka melihat orang lain sedih",
    "tidak suka membuat orang lain kecewa",
    "tidak suka menyakiti orang lain",
    "tidak suka menyakiti perasaan orang",
    "tidak suka menyalahkan orang",
    "tidak tega melihat orang kesulitan",

    "damai","berdamai","suasana damai","harmonis","keharmonisan",
    "menghindari konflik","menghindari pertengkaran","menghindari perselisihan",
    "tidak suka konflik","tidak suka pertengkaran",
    "lebih memilih berdamai","lebih memilih kompromi",
    "lebih memilih mengalah","mengalah daripada konflik",
    "mengalah daripada bertengkar",

    "kerja sama","bekerja sama","kolaboratif","kebersamaan",
    "mudah bekerja sama","senang bekerja sama",
    "lebih memilih kerja sama daripada bersaing",

    "menjaga hubungan baik","menjaga perasaan orang",
    "menjadi pendengar yang baik","memahami perasaan orang",
    "memahami sudut pandang orang lain",
    "mudah tersentuh","mudah memahami perasaan",
    "selalu berusaha bersikap baik",
    "selalu berusaha bersikap adil",
    "senang membantu tanpa mengharapkan balasan",

    "suasana nyaman","suasana rukun","suasana harmonis",
    "menciptakan suasana positif","menjaga suasana nyaman",
    "tidak enak hati","tidak enak menolak bantuan"
]
# ====== INTROSPECTION / ANALYTICAL / OCEAN
INTROSPECTION = [
    "merenung","berpikir","refleksi","evaluasi","mengamati","menganalisis","mengingat","menyadari",
    "kontemplasi","renungan","introspeksi","memikirkan","mencermati","merenungkan","berandai-andai",
    "filosofis","menghayati","meneliti","mendalami","menafsirkan","merenungi","observasi","perenungan",
    "pemikiran mendalam","kritis","analisis","evaluasi","menganalisis","mempertimbangkan","memeriksa","menguji",
    "analitis","rasional","logis","problem solving","menganalisis data",
    "observasi mendalam","pemikiran kritis","refleksi mendalam","analisis terperinci"
]

# ====== ACHIEVEMENT / DISCIPLINE / PRODUCTIVITY
ACHIEVEMENT = [
    "disiplin","tekun","bertanggung jawab","menyelesaikan","goal","target","berusaha","gigih",
    "produktif","rajin","fokus","komitmen","dedikasi","berprestasi","inisiatif","teliti","rapi",
    "terorganisir","mengikuti aturan","persisten","berorientasi hasil","bertekad","kemauan keras",
    "capai target","usaha maksimal","hasil maksimal","hasil optimal","pencapaian","goal oriented",
    "proaktif","inisiatif","mandiri","berinisiatif","work hard","determinasi",
    "menyelesaikan tugas tepat waktu","produktif tinggi","komitmen penuh","mencapai milestone","berorientasi prestasi"
]
COLLABORATION = [
    "bekerja sama","teamwork","kolaborasi","bersama","kooperatif","mendukung tim",
    "gotong royong","tim","kerja tim","kerjasama","saling membantu"
]
# =========================
# EXTREME NEGATIVE / SELF-HARM / SUICIDAL
# =========================
EXTREME_NEGATIVE = [
    "ingin mati", "rasanya ingin menyerah", "mati saja", "tidak ingin hidup", 
    "putus asa ingin mati", "bunuh diri", "tidak tahan hidup", "ingin bunuh diri", 
    "akhiri hidupku", "sudah tidak kuat", "tidak sanggup lagi", "lepaskan nyawa"
]

# Variants imbuhan sederhana
EXTREME_NEGATIVE += [f"ter{w}" for w in ["tekan","putus asa","stres","sakit"]]
EXTREME_NEGATIVE += [f"me{w}" for w in ["mati","bunuh","menyerah"]]



# ==========================
# OCEAN ADJUSTMENT REFINED
# ==========================
from collections import Counter
import re
import re, math
from collections import Counter

OCEAN_KEYS = ["O", "C", "E", "A", "N"]

# ==========================
# MAIN ADJUST FUNCTION
# ==========================
def adjust_ocean_by_keywords(scores: dict, text: str):
    adjusted = scores.copy()
    text_l = text.lower()
    counter = Counter(re.findall(r'\w+', text_l))

    # Flag kondisi mental berat
    has_mental_unstable = any(p in text_l for p in MENTAL_UNSTABLE_N)

    # ==========================
    # WORD-BASED KEYWORDS
    # ==========================
    for group, keywords in WORD_GROUPS.items():
        for word in keywords:
            if word in counter:
                f = math.log1p(counter[word])

                # Skip introspection jika mental tidak stabil
                if group == "INTROSPECTION" and has_mental_unstable:
                    continue

                for trait, weight in KEYWORD_TRAIT_MAP[group].items():
                    adjusted[trait] += weight * f

    # ==========================
    # PHRASE-BASED KEYWORDS
    # ==========================
    for group, phrases in PHRASE_GROUPS.items():
        for phrase in phrases:
            if phrase in text_l:
                for trait, weight in KEYWORD_TRAIT_MAP[group].items():
                    adjusted[trait] += weight

                # alert khusus
                if group in ["EXTREME_NEGATIVE", "MENTAL_UNSTABLE_N"]:
                    adjusted["EXTREME_ALERT"] = True

    # ==========================
    # CLAMP FINAL
    # ==========================
    for k in OCEAN_KEYS:
        adjusted[k] = round(min(5.0, max(1.0, adjusted[k])), 3)

    dominant = max(OCEAN_KEYS, key=lambda k: adjusted[k])
    return dominant, adjusted

KEYWORD_TRAIT_MAP = {
    # Sosial
    "NEGATIVE_SOCIAL": {"E": -0.2, "N": 0.4, "A": -0.1},
    "POSITIVE_SOCIAL": {"A": 0.6, "E": 0.3, "N": -0.05},
    "COLLABORATION": {"A": 0.8, "E": 0.5},

    # Emosi
    "EMO_POSITIVE": {"E": 0.3},
    "EMO_NEGATIVE": {"N": 0.35},
    "ANGER_EMO": {"N": 0.5},
    "SAD_EMO": {"N": 0.3},
    "ANXIETY_EMO": {"N": 0.4, "E": -0.1},

    # Kepribadian inti
    "INTROSPECTION": {"O": 0.4},
    "ACHIEVEMENT": {"C": 0.5},
    "DISCIPLINE_C": {"C": 0.8},   # ← DISIPLIN MURNI C
    "TRUST": {"A": 0.5},
    "EMPATHY_HARMONY_A": {"A": 0.7},

    # Ekstroversi
    "EXTRAVERSION_E": {"E": 0.7, "A": 0.4},
    "E_SOCIAL_DEPENDENCY": {"E": 0.6, "A": 0.3},

    # Ekstrem
    "EXTREME_NEGATIVE": {"N": 1.2, "E": -0.2},
    "MENTAL_UNSTABLE_N": {"N": 2.5},  # ← hanya N
}


# ==========================
# EMOTIONAL KEYWORD ADJUSTMENT REFINED
# ==========================
def apply_emotional_keyword_adjustment(text: str, scores: dict):
    adjusted = scores.copy()
    counter = Counter(re.findall(r'\w+', text.lower()))

    # ANGER → kuatkan N, tidak ubah O
    for word in ANGER_EMO:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.5 * f

    # SADNESS → naikkan N & sedikit O
    for word in SAD_EMO:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.3 * f
            adjusted["O"] += 0.05 * f

    # ANXIETY → naikkan N, turunkan E sedikit
    for word in ANXIETY_EMO:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.35 * f
            adjusted["E"] -= 0.1 * f

     # Social, Achievement, Trust, Relationship
    for group_name, keywords in {
        "NEGATIVE_SOCIAL": NEGATIVE_SOCIAL,
        "POSITIVE_SOCIAL": POSITIVE_SOCIAL,
        "EMO_POSITIVE": EMO_POSITIVE,
        "EMO_NEGATIVE": EMO_NEGATIVE,
        "INTROSPECTION": INTROSPECTION,
        "ACHIEVEMENT": ACHIEVEMENT,
        "TRUST": TRUST,
        "RELATIONSHIP_AFFECTION": RELATIONSHIP_AFFECTION,
        "COLLABORATION": COLLABORATION,
        "CREATIVE_DISCUSSION_A": CREATIVE_DISCUSSION_A,
        "DISCIPLINE_C": DISCIPLINE_C,
        "EXTRAVERSION_E": EXTRAVERSION_E,
        "E_SOCIAL_DEPENDENCY": E_SOCIAL_DEPENDENCY,
        "EMPATHY_HARMONY_A": EMPATHY_HARMONY_A,
        "MENTAL_UNSTABLE_N": MENTAL_UNSTABLE_N,
    }.items():
        for word in keywords:
            if word in counter:
                f = math.log(1 + counter[word])
                for trait, weight in KEYWORD_TRAIT_MAP.get(group_name, {}).items():
                    adjusted[trait] += weight * f

    # Clamp ke skala 1–5
    for k in ["O","C","E","A","N"]:  # jangan clamp EXTREME_ALERT
        adjusted[k] = round(min(5.0, max(1.0, adjusted[k])), 3)

    return adjusted

def determine_dominant_trait(scores, text):
    # Hitung E/A/N untuk konteks sosial
    social_hits = sum(1 for w in POSITIVE_SOCIAL+COLLABORATION if w in text.lower())
    emo_hits = sum(1 for w in EMO_POSITIVE if w in text.lower())

    # Jika banyak kata kolaborasi → dominan A
    if social_hits >= 1:
        return "A"
    if emo_hits >= 1:
        return "E"
    return max(scores, key=scores.get)

# ==========================
# HIGHLIGHT
# ==========================
def highlight_keywords_in_text(text: str, evidence: dict):
    tokens = re.findall(r'\w+|\W+', text)
    highlights = set()

    for items in evidence.values():
        for e in items:
            highlights.update([t.lower() for t in e["matched_tokens"]])

    result = ""
    for t in tokens:
        result += f"<mark>{t}</mark>" if t.lower() in highlights else t
    return result

# ==========================
# SUPER EXPLANATION (UPDATE EXTREME ALERT)
# ==========================
def extract_keywords(text, top_n=5):
    return [w for w,_ in Counter(re.findall(r'\w+', text.lower())).most_common(top_n)]

def generate_explanation_suggestion_super(text, adjusted, evidence):
    dominant = max(adjusted, key=adjusted.get)
    words = extract_keywords(text)
    snippet = ", ".join(words[:3])

    # Peringatan jika kalimat ekstrem
    if adjusted.get("EXTREME_ALERT"):
        explanation = (
            f"⚠ Kalimat ini mengandung indikasi emosional ekstrem / suicidal. "
            f"Kecenderungan trait {dominant} tetap terlihat, tetapi terdapat risiko tinggi. "
            f"Kata-kata seperti {snippet} menunjukkan hal tersebut."
        )
        suggestion = (
            f"Sangat disarankan untuk segera memperhatikan kondisi ini dan memberikan dukungan psikologis. "
            f"Mengamati kata-kata seperti {snippet} dapat membantu mencegah risiko lebih lanjut."
        )
    else:
        explanation = (
            f"Kalimat ini menunjukkan kecenderungan {dominant} karena kata-kata seperti {snippet} menandai pola tersebut."
        )
        suggestion = (
            f"Mengamati dan menindaklanjuti hal seperti {snippet} dapat membantu mengoptimalkan trait {dominant}."
        )

    return explanation, suggestion

def determine_dominant_contextual(adjusted, evidence):
    sorted_traits = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
    top_trait, top_score = sorted_traits[0]

    # Hitung jumlah bukti sosial / positif
    social_hits = len(evidence.get("POSITIVE_SOCIAL", []))
    emo_hits = len(evidence.get("EMO_POSITIVE", []))

    if social_hits >= 2:
        return "E"
    if emo_hits >= 2:
        return "A"

    return top_trait

PERSONA_RULES = [
    # ================= EMOSI NEGATIF =================
    (
        "Cemas & Pikiran Tidak Tenang",
        lambda s: s["N"] >= 3.6,
        "mudah gugup, sulit merasa tenang, dan sering diliputi pikiran negatif"
    ),
    (
        "Overthinking Emosional",
        lambda s: s["N"] >= 3.5 and s["O"] <= 3.2,
        "pikiran sulit dikendalikan, sering cemas, dan kurang stabil secara emosi"
    ),
    (
        "Rentan Stres",
        lambda s: s["N"] >= 3.4 and s["C"] <= 3.0,
        "mudah tertekan oleh situasi dan membutuhkan manajemen stres yang baik"
    ),
    (
        "Empatik & Penjaga Harmoni",
        lambda s: s["A"] >= 3.7,
        "peduli, mudah memahami perasaan orang lain, dan mengutamakan keharmonisan"
    ),
    (
        "Pendamai Alami",
        lambda s: s["A"] >= 3.6 and s["N"] <= 3.2,
        "menghindari konflik, memilih kompromi, dan menciptakan suasana damai"
    ),
    (
        "Kolaborator Hangat",
        lambda s: s["A"] >= 3.5 and s["E"] >= 3.0,
        "mudah bekerja sama, suportif, dan menjaga hubungan interpersonal"
    ),
    (
        "Pendengar yang Baik",
        lambda s: s["A"] >= 3.4 and s["O"] <= 3.2,
        "lebih fokus pada perasaan manusia daripada perdebatan ide"
    ),
    (
        "Sosial Aktif",
        lambda s: s["E"] >= 3.6,
        "energik, aktif berinteraksi, dan merasa hidup saat bersama orang lain"
    ),
    (
        "Penggerak Diskusi",
        lambda s: s["E"] >= 3.5 and s["O"] >= 3.0,
        "suka berdiskusi, memancing ide, dan menjaga dinamika percakapan"
    ),
    (
        "Team Energizer",
        lambda s: s["E"] >= 3.4 and s["C"] >= 3.0,
        "menghidupkan suasana tim dan mendorong kolaborasi aktif"
    ),
    (
        "Ekstrovert Sosial",
        lambda s: s["E"] >= 3.8 and s["N"] <= 3.0,
        "percaya diri, mudah bergaul, dan nyaman di lingkungan sosial"
    ),
    (
        "Disiplin & Bertanggung Jawab",
        lambda s: s["C"] >= 3.8,
        "terstruktur, konsisten, fokus pada kualitas, dan dapat diandalkan"
    ),
    (
        "Perfeksionis Terstruktur",
        lambda s: s["C"] >= 3.6 and s["N"] <= 3.2,
        "menjaga standar tinggi, rapi, dan tidak mentoleransi pekerjaan asal-asalan"
    ),
    (
        "Manajer Tugas Andal",
        lambda s: s["C"] >= 3.5 and s["E"] >= 3.0,
        "mampu mengatur pekerjaan, waktu, dan tanggung jawab secara efektif"
    ),
    (
        "Pekerja Sistematis",
        lambda s: s["C"] >= 3.4 and s["O"] <= 3.2,
        "lebih nyaman dengan rencana jelas, alur kerja pasti, dan target terukur"
    ),
    (
        "Sensitif Emosional",
        lambda s: s["N"] >= 3.6 and s["N"] >= s["O"] + 0.2,
        "emosional, peka, dan mudah terpengaruh suasana"
    ),
    (
        "Kolaborator Intelektual",
        lambda s: s["A"] >= 3.6 and s["O"] >= 3.4,
        "terbuka terhadap ide, menyukai diskusi mendalam, dan nyaman berkolaborasi lintas perspektif"
    ),
    (
        "Pemikir Terbuka & Solutif",
        lambda s: s["A"] >= 3.5 and s["O"] >= 3.5 and s["C"] >= 3.0,
        "menggabungkan empati, kreativitas, dan logika untuk menemukan solusi bersama"
    ),
    (
        "Idealis Kolaboratif",
        lambda s: s["A"] >= 3.7 and s["O"] >= 3.6 and s["N"] <= 3.2,
        "berorientasi nilai, menyukai dialog intelektual, dan membangun inovasi secara kolektif"
    ),

    (
        "Tempramental",
        lambda s: s["N"] >= 4.0,
        "cepat marah, impulsif, dan reaktif terhadap frustrasi"
    ),
    (
        "Cemas & Overthinking",
        lambda s: s["N"] >= 3.5 and s["E"] <= 3.0,
        "mudah khawatir, berpikir berlebihan, dan gelisah"
    ),
    (
        "Sedih / Melankolis",
        lambda s: s["N"] >= 3.2 and s["O"] >= 3.0 and s["E"] <= 3.2,
        "sering merenung, mudah merasa kehilangan, dan introspektif"
    ),

    # ================= EMOSI POSITIF =================
    (
        "Romantis",
        lambda s: s["A"] >= 3.4 and s["A"] >= s["O"] + 0.2,
        "hangat, penuh afeksi, dan berorientasi hubungan"
    ),
    (
        "Ramah Sosial",
        lambda s: s["E"] >= 3.5 and s["A"] >= 3.2,
        "ceria, mudah bergaul, dan menyukai interaksi sosial"
    ),
    (
        "Empatik",
        lambda s: s["A"] >= 3.5 and s["N"] <= 3.2,
        "peduli, memahami perasaan orang lain, dan suportif"
    ),
    (
        "Kritik & Kritis",
        lambda s: s["O"] >= 3.7 and s["C"] >= 3.2,
        "analitis, kritis, dan memperhatikan detail"
    ),
    (
        "Visioner Kreatif",
        lambda s: s["O"] >= 3.7 and s["O"] >= s["A"] + 0.2,
        "imajinatif, reflektif, dan terbuka terhadap ide baru"
    ),
    (
        "Inovator",
        lambda s: s["O"] >= 3.5 and s["C"] >= 3.0 and s["E"] >= 3.0,
        "selalu mencari cara baru, kreatif, dan berpikir out-of-the-box"
    ),

    # ================= PENCAPAIAN & DISCIPLIN =================
    (
        "Perfeksionis",
        lambda s: s["C"] >= 3.6 and s["C"] >= s["N"] + 0.2,
        "terstruktur, disiplin, dan berorientasi pencapaian"
    ),
    (
        "Ambisius",
        lambda s: s["C"] >= 3.5 and s["O"] >= 3.5,
        "berorientasi tujuan, proaktif, dan berinisiatif"
    ),
    (
        "Gigih & Persisten",
        lambda s: s["C"] >= 3.4 and s["N"] <= 3.2,
        "konsisten, tidak mudah menyerah, dan berdedikasi"
    ),
    (
        "Pragmatis",
        lambda s: s["C"] >= 3.2 and s["E"] >= 3.2,
        "praktis, realistis, dan fokus pada hasil"
    ),

    # ================= KOLABORATOR =================
    (
        "Kolaborator",
        lambda s: s["A"] >= 3.5 and s["E"] >= 3.2 and s["C"] >= 3.0,
        "mampu bekerja sama, mendukung tim, dan membangun harmoni"
    ),
    (
        "Mediator",
        lambda s: s["A"] >= 3.3 and s["N"] <= 3.2 and s["E"] >= 3.0,
        "menjembatani konflik, tenang, dan diplomatis"
    ),
    (
        "Pemimpin Visioner",
        lambda s: s["O"] >= 3.6 and s["C"] >= 3.5 and s["E"] >= 3.2,
        "mengambil inisiatif, memimpin tim, dan strategis"
    ),

    # ================= KEPRIBADIAN SEIMBANG =================
    (
        "Seimbang",
        lambda s: 2.8 <= s["O"] <= 3.5 and 2.8 <= s["C"] <= 3.5 and 2.8 <= s["E"] <= 3.5 and 2.8 <= s["A"] <= 3.5 and 2.8 <= s["N"] <= 3.5,
        "adaptif, fleksibel, dan tidak ekstrem pada satu trait"
    )
]
def generate_global_conclusion(avg, dominant):
    O, C, E, A, N = avg["O"], avg["C"], avg["E"], avg["A"], avg["N"]

    # ================= KESIMPULAN =================
    conclusion = (
        f"Secara keseluruhan, hasil analisis menunjukkan bahwa trait kepribadian "
        f"yang paling dominan adalah {dominant}. Individu ini cenderung "
    )

    if dominant == "O":
        conclusion += "memiliki tingkat keterbukaan tinggi terhadap ide baru, reflektif, dan kreatif."
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
        conclusion += " Namun terdapat kecenderungan emosi negatif yang cukup tinggi."

    # ================= SARAN =================
    suggestion = "Disarankan untuk "

    if dominant == "C":
        suggestion += (
            "memanfaatkan kemampuan perencanaan dan kedisiplinan dalam pekerjaan atau studi, "
            "namun tetap melatih fleksibilitas agar tidak terlalu kaku."
        )
    elif dominant == "O":
        suggestion += (
            "menyalurkan kreativitas ke aktivitas produktif seperti riset, inovasi, dan eksplorasi ide baru."
        )
    elif dominant == "E":
        suggestion += (
            "mengoptimalkan kemampuan komunikasi dan kepemimpinan dalam kerja tim, "
            "serta melatih kemampuan refleksi diri."
        )
    elif dominant == "A":
        suggestion += (
            "mempertahankan sikap empati sambil belajar bersikap lebih tegas dalam pengambilan keputusan."
        )
    elif dominant == "N":
        suggestion += (
            "melatih regulasi emosi melalui manajemen stres, mindfulness, atau journaling secara rutin."
        )

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
