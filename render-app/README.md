# XRayGuard

Evidence-grounded chest X-ray report drafting with a clinician verification workflow.
Research prototype for a capstone project — not for clinical use.

## Pages
- **Analyze** – upload a frontal chest X-ray, get scores for the 14 NIH ChestX-ray14 findings, finding tiers, a region map, a draft report, and a radiologist sign-off form.
- **Evaluation** – results of the trained DenseNet-121 model (NIH 10k subset) and a summary of clinician reviews.
- **Our approach** – pipeline, related work, scope and limitations.

On this deployment, image scores come from a vision model API (free Gemini by default, or Claude if ANTHROPIC_API_KEY is set) following the XRayGuard pipeline; the Evaluation numbers come from the trained DenseNet-121.

## Run on your computer (easiest)
- **Windows:** double-click `start.bat`
- **Mac / Linux:** run `./start.sh`

The first run installs everything (1-2 minutes) and asks for your free Gemini API key once (get it at https://aistudio.google.com/apikey, no card) (saved in `.env`).
Your browser opens http://localhost:8000. Keep the window open while you use it.

To change the key later, delete `.env` and run the script again.

## Run locally (manual)
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...
uvicorn app:app --reload
```
Open http://localhost:8000

## Deploy on Render
1. New → Blueprint → select this repo (uses `render.yaml`).
2. Set `GEMINI_API_KEY` when prompted.
3. Deploy. Health check: `/health`.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Free key from Google AI Studio (used if set). Never commit it. |
| `ANTHROPIC_API_KEY` | – | Alternative paid provider. |
| `MODEL` | `gemini-2.5-flash` (Gemini) / `claude-sonnet-5-5` | Model used for image scoring |
| `LIMIT_PER_HOUR` | `30` | Requests per IP per hour |
