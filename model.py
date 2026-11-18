"""
XRayGuard – DenseNet121-based chest X-ray classifier (14 findings).
Trained on the NIH ChestX-ray14 dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# Human-readable aliases for report text
WORD = {
    "Lung Opacity": "opacity",
    "Lung Lesion": "lung lesion",
    "Pleural Other": "pleural thickening",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
    "Support Devices": "support devices",
    "No Finding": "no acute findings",
}

# NIH label → unified label mapping
NIH_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
    "Emphysema": "Lung Opacity",
    "Fibrosis": "Lung Opacity",
    "Hernia": "Enlarged Cardiomediastinum",
    "Infiltration": "Lung Opacity",
    "Mass": "Lung Lesion",
    "Nodule": "Lung Lesion",
    "Pleural_Thickening": "Pleural Other",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


class XRayGuardModel(nn.Module):
    """DenseNet121 with multi-label sigmoid classification head."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        in_features = self.densenet.classifier.in_features  # 1024
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.labels = LABELS

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for DenseNet121."""

    def __init__(self, model: XRayGuardModel):
        self.model = model

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        with torch.enable_grad():
            # Clone input and enable gradient tracking on features
            features = self.model.forward_features(x)
            features = features.clone().detach().requires_grad_(True)
            
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            logits = self.model.densenet.classifier(out)

            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

            gradients = features.grad
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling over spatial dims
            cam = (weights * features).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()

            # Normalize to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
            return cam
