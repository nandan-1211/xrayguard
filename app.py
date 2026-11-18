"""
XRayGuard – Flask web server.
Serves the UI and provides API endpoints for X-ray analysis.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

# Global predictor (lazy-loaded or pre-warmed)
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        from predict import XRayPredictor
        work_dir = app.config.get("WORK_DIR", os.path.join(BASE_DIR, "weights"))
        predictor = XRayPredictor(work_dir=work_dir)
    return predictor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        pred = get_predictor()
        return jsonify({
            "status": "ok",
            "demo_mode": pred.demo_mode,
            "device": str(pred.device),
            "model_loaded": True,
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": str(exc),
        }), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept an X-ray image and return a structured report."""
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        # Read image
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Save uploaded image for preview / caching
        filename = f"upload_{int(time.time()*1000)}.png"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        pil_img.save(save_path, "PNG")

        # Run prediction
        pred = get_predictor()
        result = pred.predict(pil_img)

        # Build response
        response = {
            "success": True,
            "time_ms": round((time.time() - start) * 1000, 1),
            "image_url": f"/static/uploads/{filename}",
            "demo_mode": result["demo_mode"],
            "report": {
                "findings": result["findings"],
                "impression": result["impression"],
            },
            "probabilities": {
                k: round(v, 4) for k, v in result["probabilities"].items()
            },
            "findings_present": result["findings_present"],
            "findings_possible": result["findings_possible"],
            "findings_absent": result["findings_absent"],
            "tiers": result["tiers"],
            "heatmap": result["heatmap"],  # base64 PNG or None
        }
        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {str(exc)}"}), 500


@app.route("/api/sample-report")
def sample_report():
    """Return a sample report for demo purposes."""
    return jsonify({
        "success": True,
        "demo_mode": True,
        "report": {
            "findings": (
                "There is cardiomegaly (confidence: 87.3%). "
                "There is pleural effusion (confidence: 72.1%). "
                "There may be atelectasis (confidence: 45.6%). "
                "No pneumothorax or consolidation."
            ),
            "impression": "Cardiomegaly. Pleural Effusion. Possible atelectasis.",
        },
        "probabilities": {
            "Atelectasis": 0.456,
            "Cardiomegaly": 0.873,
            "Consolidation": 0.089,
            "Edema": 0.234,
            "Enlarged Cardiomediastinum": 0.123,
            "Fracture": 0.034,
            "Lung Lesion": 0.067,
            "Lung Opacity": 0.312,
            "No Finding": 0.045,
            "Pleural Effusion": 0.721,
            "Pleural Other": 0.056,
            "Pneumonia": 0.145,
            "Pneumothorax": 0.023,
            "Support Devices": 0.189,
        },
        "findings_present": ["Cardiomegaly", "Pleural Effusion"],
        "findings_possible": ["Atelectasis"],
        "findings_absent": [
            "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
        ],
        "tiers": {
            "present": [
                {"finding": "Cardiomegaly", "confidence": 87.3, "tier": "Present"},
                {"finding": "Pleural Effusion", "confidence": 72.1, "tier": "Present"},
            ],
            "possible": [
                {"finding": "Atelectasis", "confidence": 45.6, "tier": "Possible"},
            ],
            "absent": [],
        },
        "heatmap": None,
    })


def main():
    parser = argparse.ArgumentParser(description="XRayGuard web server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--work_dir", type=str, default=os.path.join(BASE_DIR, "weights"),
                        help="Directory containing model weights")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.config["WORK_DIR"] = args.work_dir
    print(f"\n{'='*50}")
    print(f"  XRayGuard v1.0")
    print(f"  Server: http://localhost:{args.port}")
    print(f"  Model dir: {os.path.abspath(args.work_dir)}")
    print(f"{'='*50}\n")

    # Warm up model so first user request is instantaneous
    print("Pre-warming model and weights...")
    try:
        get_predictor()
        print("Model pre-warmed successfully!")
    except Exception as e:
        print(f"Warning during pre-warm: {e}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
