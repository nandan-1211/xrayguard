import os
import re
import time
from collections import defaultdict, deque

import anthropic
import httpx


def _load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]
PROMPT = """You are scoring a chest radiograph for a research prototype. Look at the attached image and output ONLY JSON.
First decide if it is a frontal chest radiograph. Then give a score from 0 to 1 for how strongly each finding is visible: {FINDINGS}.
Be conservative: most findings are absent on most films, so most scores should be low. For any finding scored 0.3 or higher, give the region where it is visible as a box in image coordinates (fractions 0-1 of width and height: x, y, w, h).
JSON shape: {"is_chest_xray":bool,"scores":[{"name":string,"score":number,"box":{"x":number,"y":number,"w":number,"h":number}|null}]}"""
PROVIDER = "gemini" if os.getenv("GEMINI_API_KEY") else "anthropic"
MODEL = os.getenv("MODEL") or ("gemini-2.5-flash" if PROVIDER == "gemini" else "claude-sonnet-5-5")
LIMIT_PER_HOUR = int(os.getenv("LIMIT_PER_HOUR", "30"))
MAX_IMAGE_B64 = 8_000_000

app = FastAPI(title="XRayGuard")
client = None
hits = defaultdict(deque)


def get_client():
    global client
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "Server has no API key. Set GEMINI_API_KEY (free) or ANTHROPIC_API_KEY.")
        client = anthropic.Anthropic()
    return client


class ScoreRequest(BaseModel):
    images: list[str]
    findings: list[str] | None = None


@app.get("/health")
def health():
    key = os.getenv("GEMINI_API_KEY") if PROVIDER == "gemini" else os.getenv("ANTHROPIC_API_KEY")
    return {"status": "ok", "provider": PROVIDER, "model": (_gemini_model or MODEL) if PROVIDER == "gemini" else MODEL, "key_configured": bool(key)}


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_gemini_model = None


def _gemini_headers():
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip(), "Content-Type": "application/json"}


def gemini_models():
    """All free Flash models this key can use that read images, best first."""
    try:
        r = httpx.get(f"{GEMINI_BASE}/models", params={"pageSize": 200}, headers=_gemini_headers(), timeout=30)
    except httpx.HTTPError:
        return []
    if r.status_code != 200:
        return []
    names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    bad = ("image", "tts", "audio", "live", "embedding", "thinking", "exp", "robotics", "computer", "gemma", "pro", "omni")
    flash = [n for n in names if "flash" in n and not any(b in n for b in bad)]

    def rank(n):
        nums = [int(x) if x.isdigit() else 0 for x in re.findall(r"\d+", n)]
        return ("lite" not in n, "preview" not in n, nums)

    flash.sort(key=rank, reverse=True)
    return flash


def _gemini_call(model, image_b64, prompt):
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}, {"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    try:
        return httpx.post(f"{GEMINI_BASE}/models/{model}:generateContent", headers=_gemini_headers(), json=body, timeout=90)
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach the Gemini service. Check your internet connection.")


def ask_gemini(image_b64, prompt):
    global _gemini_model
    first = _gemini_model or os.getenv("MODEL") or "gemini-2.5-flash"
    tried, candidates, listed = [], [first], False
    r = None
    while candidates:
        model = candidates.pop(0)
        if model in tried:
            continue
        tried.append(model)
        for attempt in range(2):
            r = _gemini_call(model, image_b64, prompt)
            if r.status_code not in (500, 503) or attempt == 1:
                break
            print(f"[gemini] '{model}' busy ({r.status_code}) - retrying in 3 s", flush=True)
            time.sleep(3)
        if r.status_code == 200:
            _gemini_model = model
            break
        busy = r.status_code in (429, 500, 503)
        missing = r.status_code == 404 or (r.status_code == 400 and "model" in r.text.lower() and "not" in r.text.lower())
        if not (busy or missing):
            break
        if not listed:
            listed = True
            candidates += [m for m in gemini_models() if m not in tried]
        if candidates:
            print(f"[gemini] '{model}' {'busy' if busy else 'unavailable'} ({r.status_code}) - trying '{candidates[0]}'", flush=True)
        if len(tried) >= 6:
            break
    if r.status_code in (500, 503):
        raise HTTPException(503, "Google's free Gemini models are busy right now. Wait a minute and click Generate report again.")
    if r.status_code == 429:
        raise HTTPException(429, "Free Gemini quota reached. Wait a minute (or until tomorrow) and try again.")
    if r.status_code in (401, 403) or (r.status_code == 400 and "api key" in r.text.lower()):
        raise HTTPException(502, "Gemini rejected the API key. Create a new key at aistudio.google.com/apikey, delete the .env file and run start.bat again.")
    if r.status_code == 404:
        raise HTTPException(502, "No Gemini model is available for this key. Check the key at aistudio.google.com/apikey.")
    if r.status_code != 200:
        print("[gemini]", r.status_code, r.text[:300])
        raise HTTPException(502, f"Gemini service error ({r.status_code}).")
    try:
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        raise HTTPException(502, "Gemini returned no answer for this image. Try another image.")


@app.post("/api/score")
def score(req: ScoreRequest, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    now, q = time.time(), hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= LIMIT_PER_HOUR:
        raise HTTPException(429, "Too many requests. Try again later.")
    if len(req.images) != 1 or len(req.images[0]) > MAX_IMAGE_B64:
        raise HTTPException(400, "Send exactly one image up to 6 MB.")
    q.append(now)
    prompt = PROMPT.replace("{FINDINGS}", ", ".join(FINDINGS))
    if PROVIDER == "gemini":
        return {"text": ask_gemini(req.images[0], prompt)}
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": req.images[0]}},
               {"type": "text", "text": prompt}]
    try:
        msg = get_client().messages.create(model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": content}])
    except anthropic.RateLimitError:
        raise HTTPException(429, "The AI service is busy. Try again in a minute.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"AI service error ({e.status_code}).")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the AI service.")
    return {"text": "".join(b.text for b in msg.content if b.type == "text")}


PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
PAGE_VERSION = "2026-11-19 - Analyze + Our approach (no Evaluation page)"
NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"}
print(f"\n  Serving page : {PAGE}\n  Page version : {PAGE_VERSION}\n", flush=True)


@app.get("/")
def index():
    return FileResponse(PAGE, headers=NO_CACHE)


@app.get("/version")
def version():
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    return {"page_version": PAGE_VERSION, "page_file": PAGE,
            "has_evaluation_tab": ">Evaluation</a>" in html, "has_results_table": "<th>MSE</th>" in html}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")), name="static")
