import os
from dotenv import load_dotenv

load_dotenv()

# Twitter
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")

# Model
HF_REPO = "sapadev13/sapa_ocean_id"
ONTOLOGY_CSV = "ontology_clean.csv"
ONTOLOGY_EMB = "ontology_embeddings.pt"
DEVICE = "cpu"
MAX_LEN = 256

# Session
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-secret")
