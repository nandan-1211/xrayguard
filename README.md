# 🏥 XRayGuard – Chest X-ray Report Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DenseNet-121](https://img.shields.io/badge/Architecture-DenseNet--121-green.svg)](https://arxiv.org/abs/1608.06993)
[![Dataset: NIH ChestX-ray14](https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-orange.svg)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**XRayGuard** is an end-to-end chest X-ray analysis and automated radiology report generation platform. Powered by **DenseNet121** trained on the **NIH ChestX-ray14** dataset, it analyzes frontal chest radiographs across **14 distinct cardiopulmonary conditions**, generates structured clinical radiology reports (Findings + Impression), computes **GradCAM attention heatmaps**, and provides a web interface.

---

## 🌟 Key Features

- **14 Cardiopulmonary Findings**: Multi-label classification covering *Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices, and No Finding*.
- **Structured Radiology Reports**: Generates formal clinical radiology reports divided into structured *Findings* (with explicit key negative denials) and prioritized *Impressions*.
- **GradCAM Explainability**: Visual attention heatmaps highlight the anatomical regions triggering the model's highest predictions.
- **Calibrated Confidence**: Calibrated operating thresholds to avoid false alarms while maintaining clinical specificity.
- **Modern Responsive Web UI**: Drag-and-drop X-ray upload, instant preview, confidence score meters, 1-click sample test, copy/download report options.
- **Ready for Deployment**: Includes Dockerfile, Procfile, and simple Flask backend for quick deployment to cloud providers (Render, Railway, Hugging Face Spaces, GCP).

---

## 🚀 Quick Start (Local Run)

### 1. Clone the repository
```bash
git clone https://github.com/nandan-1211/xrayguard.git
cd xrayguard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py --port 5000
```
Open your browser and navigate to **`http://localhost:5000`**.

---

## 📁 Project Structure

```
xrayguard/
├── app.py                      # Flask web server & API endpoints
├── model.py                    # DenseNet121 architecture + GradCAM hook implementation
├── predict.py                  # Calibrated inference engine & report synthesizer
├── prepare_data.py             # NIH dataset extractor & manifest builder
├── train.py                    # Full training pipeline with BCE loss & temperature scaling
├── XRayGuard_Train_Colab.ipynb # Free GPU training notebook for Google Colab
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization for cloud deployment
├── Procfile                    # PaaS process configuration (Render/Railway/Heroku)
├── templates/
│   └── index.html              # Frontend user interface
├── static/
│   ├── css/style.css           # Styling
│   ├── js/app.js               # Frontend controller & API integration
│   └── samples/                # Sample frontal chest X-ray for 1-click testing
└── weights/                    # Directory for custom model checkpoints
```

---

## 🧠 Model & Dataset

- **Backbone**: DenseNet121 feature extractor with a custom multi-label sigmoid classification head.
- **Pretrained Weights**: Uses weights trained on the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) (112,120 frontal chest X-rays from 30,805 unique patients).
- **Explainability**: Gradient-weighted Class Activation Mapping (GradCAM) overlays visual heatmaps on top of the original radiograph.

---

## 🔬 Training on NIH ChestX-ray14

You can train or fine-tune your own weights using the provided training scripts:

```bash
# Step 1: Download & prepare dataset manifest (e.g., 5000 images)
python prepare_data.py --data_dir ./data --max_images 5000 --output manifest.csv

# Step 2: Run training with class-weighted BCE loss and temperature calibration
python train.py --manifest manifest.csv --work_dir ./weights --epochs 15 --bs 16 --weighted
```

Alternatively, open `XRayGuard_Train_Colab.ipynb` directly in Google Colab to train on a free T4 GPU.

---

## 🌐 API Reference

### Health Check
`GET /api/health`
```json
{
  "status": "ok",
  "demo_mode": false,
  "device": "cpu",
  "model_loaded": true
}
```

### Predict & Generate Report
`POST /api/predict`
- **Body**: `multipart/form-data` with `image` file field.
- **Response**:
```json
{
  "success": true,
  "time_ms": 2411.3,
  "demo_mode": false,
  "report": {
    "findings": "Borderline appearance suspicious for mild focal consolidation...",
    "impression": "1. Possible mild focal consolidation; recommend correlation."
  },
  "probabilities": { "Cardiomegaly": 0.556, "Pneumonia": 0.528, ... },
  "findings_present": [],
  "findings_possible": ["Consolidation", "Lung Opacity"],
  "heatmap": "<base64_encoded_png>"
}
```

---

## ⚠️ Medical Disclaimer

XRayGuard is developed for **academic and research purposes only**. It is **not** intended for primary clinical diagnosis or medical decision-making. All radiographic findings must be interpreted and verified by a licensed medical professional / radiologist.

## Render web app

The Render-ready Gemini-backed web app lives in [render-app](render-app/). The root [render.yaml](render.yaml) deploys that service without replacing this repository's original model-backed application. Set GEMINI_API_KEY as a Render secret; do not commit .env.
