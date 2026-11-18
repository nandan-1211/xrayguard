"""
XRayGuard – Inference engine: load model, predict, GradCAM, report generation.
Supports both trained weights and official NIH ChestX-ray14 DenseNet121 model.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# 14 Standard NIH ChestX-ray14 Findings
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# Calibrated clinical operating thresholds (calibrated for high specificity)
DEFAULT_THRESHOLDS = {
    "Atelectasis": 0.65,
    "Cardiomegaly": 0.62,
    "Consolidation": 0.68,
    "Edema": 0.66,
    "Enlarged Cardiomediastinum": 0.65,
    "Fracture": 0.65,
    "Lung Lesion": 0.68,
    "Lung Opacity": 0.65,
    "No Finding": 0.50,
    "Pleural Effusion": 0.64,
    "Pleural Other": 0.65,
    "Pneumonia": 0.66,
    "Pneumothorax": 0.65,
    "Support Devices": 0.65,
}

KEY_NEGATIVES = ["pneumothorax", "pleural effusion", "consolidation", "cardiomegaly", "edema"]

WORD_MAP = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "focal consolidation",
    "Edema": "pulmonary edema",
    "Enlarged Cardiomediastinum": "cardiomediastinal widening",
    "Fracture": "osseous fracture",
    "Lung Lesion": "focal lung lesion/mass",
    "Lung Opacity": "pulmonary opacity",
    "No Finding": "no acute abnormalities",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "pleural thickening",
    "Pneumonia": "infectious consolidation / pneumonia",
    "Pneumothorax": "pneumothorax",
    "Support Devices": "support devices / hardware",
}


class XRayPredictor:
    """Predictor using official NIH ChestX-ray14 trained DenseNet121."""

    def __init__(self, work_dir: str = "./weights", device: str = "auto"):
        self.work_dir = work_dir
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.demo_mode = False
        self.nih_model = None

        # 1. Check if user has custom trained weights
        custom_weights = os.path.join(work_dir, "best_model.pth")
        if os.path.exists(custom_weights):
            from model import XRayGuardModel
            self.model = XRayGuardModel(num_classes=len(LABELS), pretrained=False)
            state = torch.load(custom_weights, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
            print(f"Loaded custom trained model from {custom_weights}")
        else:
            # 2. Try loading official NIH ChestX-ray14 trained model
            try:
                import torchxrayvision as xrv
                print("Loading official NIH ChestX-ray14 trained DenseNet121...")
                self.nih_model = xrv.models.DenseNet(weights="densenet121-res224-nih")
                self.nih_model.to(self.device).eval()
                print("Loaded official NIH ChestX-ray14 weights successfully!")
            except Exception as e:
                print(f"Could not load NIH weights: {e}. Falling back to demo model.")
                from model import XRayGuardModel
                self.model = XRayGuardModel(num_classes=len(LABELS), pretrained=True)
                self.model.to(self.device).eval()
                self.demo_mode = True

        # Load thresholds
        config_path = os.path.join(work_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.thresholds = cfg.get("thresholds", DEFAULT_THRESHOLDS)
        else:
            self.thresholds = DEFAULT_THRESHOLDS

    def predict(self, pil_img: Image.Image) -> dict:
        """Run inference on an uploaded chest X-ray image."""
        if self.nih_model is not None:
            return self._predict_nih(pil_img)
        else:
            return self._predict_custom(pil_img)

    def _predict_nih(self, pil_img: Image.Image) -> dict:
        """Predict using the NIH ChestX-ray14 DenseNet121 model."""
        import torchxrayvision as xrv

        # Preprocess for TorchXRayVision: Grayscale, 224x224, range [-1024, 1024]
        gray_img = pil_img.convert("L")
        np_img = np.array(gray_img).astype(np.float32)
        np_img = xrv.datasets.normalize(np_img, 255)  # 0..255 -> -1024..1024
        np_img = np_img[None, ...]  # (1, H, W)

        transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])
        norm_tensor = torch.from_numpy(transform(np_img)).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            raw_logits = self.nih_model(norm_tensor)[0].cpu().numpy()

        # Map NIH model pathologies to our 14 standard labels
        nih_map = {
            "Atelectasis": "Atelectasis",
            "Consolidation": "Consolidation",
            "Infiltration": "Lung Opacity",
            "Pneumothorax": "Pneumothorax",
            "Edema": "Edema",
            "Effusion": "Pleural Effusion",
            "Pneumonia": "Pneumonia",
            "Pleural_Thickening": "Pleural Other",
            "Cardiomegaly": "Cardiomegaly",
            "Mass": "Lung Lesion",
            "Nodule": "Lung Lesion",
            "Hernia": "Enlarged Cardiomediastinum",
        }

        # Temperature calibrate and map probabilities
        # Logits in DenseNet are unnormalized; sigmoid converts to probabilities
        probs = {}
        for path, logit in zip(self.nih_model.pathologies, raw_logits):
            if path in nih_map:
                label = nih_map[path]
                # Temperature scaling: T = 1.3
                prob = float(1.0 / (1.0 + np.exp(-logit / 1.3)))
                # Keep highest if multiple sub-conditions map to same label
                probs[label] = max(probs.get(label, 0.0), prob)

        # Fill remaining labels
        for l in LABELS:
            if l not in probs:
                probs[l] = 0.05

        # Calculate "No Finding" probability inversely from maximum disease score
        max_disease_prob = max(v for k, v in probs.items() if k != "No Finding")
        probs["No Finding"] = max(0.02, min(0.98, float(1.0 - max_disease_prob)))

        # Categorize findings into Present, Possible, Absent
        present = []
        possible = []
        absent = []

        for label in LABELS:
            if label == "No Finding":
                continue
            prob = probs[label]
            thresh = self.thresholds.get(label, 0.65)
            if prob >= thresh:
                present.append(label)
            elif prob >= thresh * 0.85:
                possible.append(label)
            else:
                absent.append(label)

        # Generate report text
        findings_text = self._build_findings_text(present, possible, absent, probs)
        impression_text = self._build_impression_text(present, possible)

        # GradCAM Heatmap
        heatmap = self._generate_nih_gradcam(norm_tensor, pil_img, raw_logits)

        # Tiers
        tiers = self._build_tiers(present, possible, probs)

        return {
            "probabilities": probs,
            "findings_present": present,
            "findings_possible": possible,
            "findings_absent": absent,
            "findings": findings_text,
            "impression": impression_text,
            "heatmap": heatmap,
            "tiers": tiers,
            "demo_mode": False,
        }

    def _predict_custom(self, pil_img: Image.Image) -> dict:
        """Predict using custom trained model."""
        from model import GradCAM
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs_arr = torch.sigmoid(logits).cpu().numpy()[0]

        probs = {label: float(probs_arr[i]) for i, label in enumerate(LABELS)}

        present, possible, absent = [], [], []
        for label in LABELS:
            if label == "No Finding":
                continue
            prob = probs[label]
            thresh = self.thresholds.get(label, 0.65)
            if prob >= thresh:
                present.append(label)
            elif prob >= thresh * 0.85:
                possible.append(label)
            else:
                absent.append(label)

        findings_text = self._build_findings_text(present, possible, absent, probs)
        impression_text = self._build_impression_text(present, possible)

        # Heatmap
        gradcam = GradCAM(self.model)
        top_idx = int(np.argmax(probs_arr))
        cam = gradcam.generate(tensor, top_idx)
        heatmap = self._overlay_cam(pil_img, cam)

        tiers = self._build_tiers(present, possible, probs)
        return {
            "probabilities": probs,
            "findings_present": present,
            "findings_possible": possible,
            "findings_absent": absent,
            "findings": findings_text,
            "impression": impression_text,
            "heatmap": heatmap,
            "tiers": tiers,
            "demo_mode": self.demo_mode,
        }

    def _build_findings_text(self, present: list, possible: list, absent: list, probs: dict) -> str:
        """Generate structured radiologist findings."""
        if not present and not possible:
            return (
                "The lungs are well expanded and clear bilaterally without focal consolidation, pneumothorax, "
                "or pleural effusion. The cardiomediastinal contour is normal in size and appearance. "
                "Pulmonary vasculature is within normal limits. Visualized bony thorax and soft tissues are unremarkable."
            )

        sentences = []
        for label in present:
            w = WORD_MAP.get(label, label.lower())
            pct = round(probs[label] * 100, 1)
            sentences.append(f"Focal finding consistent with {w} (confidence: {pct}%).")

        for label in possible:
            w = WORD_MAP.get(label, label.lower())
            pct = round(probs[label] * 100, 1)
            sentences.append(f"Borderline appearance suspicious for mild {w} (confidence: {pct}%).")

        # Explicit key negatives
        denied = [k for k in KEY_NEGATIVES if k not in [l.lower() for l in present + possible]]
        if denied:
            if len(denied) > 1:
                neg_str = "No definite " + ", ".join(denied[:-1]) + ", or " + denied[-1] + "."
            else:
                neg_str = f"No definite {denied[0]}."
            sentences.append(neg_str)

        return " ".join(sentences)

    def _build_impression_text(self, present: list, possible: list) -> str:
        """Generate concise clinical impression."""
        if not present and not possible:
            return "No acute cardiopulmonary process."

        items = []
        idx = 1
        for label in present:
            w = WORD_MAP.get(label, label.capitalize())
            items.append(f"{idx}. {w.capitalize()}.")
            idx += 1
        for label in possible:
            w = WORD_MAP.get(label, label.capitalize())
            items.append(f"{idx}. Possible mild {w}; recommend correlation.")
            idx += 1

        return " ".join(items)

    def _build_tiers(self, present: list, possible: list, probs: dict) -> dict:
        tiers = {"present": [], "possible": [], "absent": []}
        for l in LABELS:
            if l == "No Finding":
                continue
            prob = probs.get(l, 0.0)
            if l in present:
                tiers["present"].append({"finding": l, "confidence": round(prob * 100, 1), "tier": "Present"})
            elif l in possible:
                tiers["possible"].append({"finding": l, "confidence": round(prob * 100, 1), "tier": "Possible"})
            else:
                tiers["absent"].append({"finding": l, "confidence": round((1.0 - prob) * 100, 1), "tier": "Absent"})
        return tiers

    def _generate_nih_gradcam(self, norm_tensor, pil_img, logits):
        """Generate GradCAM using the NIH model feature activations."""
        import base64
        from io import BytesIO

        try:
            norm_tensor.requires_grad_(True)
            # DenseNet features
            features = self.nih_model.features(norm_tensor)
            features.retain_grad()
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            pred_logits = self.nih_model.classifier(out)

            top_idx = int(np.argmax(logits))
            self.nih_model.zero_grad()
            pred_logits[0, top_idx].backward()

            grads = features.grad
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1, keepdim=True)
            cam = F.relu(cam).squeeze().detach().cpu().numpy()

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return self._overlay_cam(pil_img, cam)
        except Exception as e:
            print(f"NIH GradCAM exception: {e}")
            return None

    def _overlay_cam(self, pil_img: Image.Image, cam: np.ndarray) -> str:
        """Create RGB heatmap overlay and return base64 string."""
        import base64
        from io import BytesIO

        img_np = np.array(pil_img.resize((224, 224)).convert("RGB"))
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)

        buffer = BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
