#!/usr/bin/env bash
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3 first: https://www.python.org/downloads/"; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up..."
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -r requirements.txt || exit 1
fi
if [ ! -f .env ]; then
  read -rp "Paste your free Gemini API key (from https://aistudio.google.com/apikey): " KEY
  echo "GEMINI_API_KEY=$KEY" > .env
fi
echo "Open http://localhost:8000  (Ctrl+C to stop)"
( sleep 2; command -v xdg-open >/dev/null && xdg-open http://localhost:8000 || command -v open >/dev/null && open http://localhost:8000 ) >/dev/null 2>&1 &
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
