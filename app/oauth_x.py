import os
import requests
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()

CLIENT_ID = os.getenv("X_CLIENT_ID")
CLIENT_SECRET = os.getenv("X_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/auth/x/callback"

AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"

SCOPES = "tweet.read users.read offline.access"

@router.get("/login", operation_id="x_login")
def login_x():
    url = (
        f"{AUTH_URL}"
        f"?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope={SCOPES}"
        f"&state=oceansystem"
        f"&code_challenge=challenge"
        f"&code_challenge_method=plain"
    )
    return RedirectResponse(url)

@router.get("/callback", operation_id="x_callback")
def callback_x(code: str):
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code": code,
        "code_verifier": "challenge",
    }

    r = requests.post(TOKEN_URL, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    token = r.json()

    return {
        "access_token": token.get("access_token"),
        "refresh_token": token.get("refresh_token")
    }
