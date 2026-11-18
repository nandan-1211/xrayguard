"""
XRayGuard – Data preparation from NIH ChestX-ray14 dataset.
Downloads / locates the dataset and creates a manifest CSV for training.

Usage:
    python prepare_data.py --data_dir ./data --max_images 5000 --output manifest.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from model import LABELS, NIH_MAP, WORD

KEY = ["pneumothorax", "pleural effusion", "consolidation", "cardiomegaly"]


def find_images(data_dir: str) -> dict:
    """Recursively find all PNG images in data_dir."""
    imgs = {}
    for p in glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True):
        imgs.setdefault(os.path.basename(p), p)
    return imgs


def find_labels_csv(data_dir: str) -> str:
    """Find the NIH Data_Entry CSV."""
    candidates = (
        glob.glob(os.path.join(data_dir, "**", "Data_Entry_2017*.csv"), recursive=True)
        + glob.glob("Data_Entry_2017*.csv")
    )
    if not candidates:
        # Try kagglehub download
        try:
            import kagglehub
            p = kagglehub.dataset_download(
                "nih-chest-xrays/data", path="Data_Entry_2017.csv"
            )
            if os.path.isfile(p):
                return p
            found = glob.glob(
                os.path.join(p, "**", "Data_Entry_2017*.csv"), recursive=True
            )
            if found:
                return found[0]
        except Exception as exc:
            print(f"kagglehub download failed: {exc}")
        raise FileNotFoundError(
            "Cannot find Data_Entry_2017*.csv. Place it in --data_dir or install kagglehub."
        )
    return candidates[0]


def build_manifest(
    data_dir: str,
    max_images: int = 5000,
    output: str = "manifest.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Build the training manifest CSV."""
    print(f"Scanning images in {data_dir} ...")
    imgs = find_images(data_dir)
    print(f"  Found {len(imgs)} images")

    csv_path = find_labels_csv(data_dir)
    print(f"  Labels CSV: {csv_path}")
    entry = pd.read_csv(csv_path)
    entry = entry[entry["Image Index"].isin(imgs)]

    if max_images and len(entry) > max_images:
        entry = entry.sample(n=max_images, random_state=seed)
        print(f"  Sampled {max_images} images")

    rows = []
    for _, x in entry.iterrows():
        raw_labels = str(x["Finding Labels"]).split("|")
        mapped = sorted({NIH_MAP[t] for t in raw_labels if t in NIH_MAP})

        row = {label: 0 for label in LABELS}
        for label in mapped:
            row[label] = 1
        row["No Finding"] = int(not mapped)

        # Generate template findings and impression
        neg = [k for k in KEY if k not in [l.lower() for l in mapped]]
        negs = (
            ("No " + ", ".join(neg[:-1]) + " or " + neg[-1] + ".")
            if len(neg) > 1
            else (f"No {neg[0]}." if neg else "")
        )

        if mapped:
            findings = (
                " ".join(
                    f"There is {WORD.get(l, l.lower())}." for l in mapped
                )
                + " "
                + negs
            )
            impression = " ".join(
                f"{WORD.get(l, l.lower()).capitalize()}." for l in mapped
            )
        else:
            findings = negs
            impression = "No acute cardiopulmonary process."

        row.update(
            img_path=imgs[x["Image Index"]],
            findings=findings.strip(),
            impression=impression,
            patient_id=int(x["Patient ID"]),
            view=x["View Position"] if x["View Position"] in ("PA", "AP") else "PA",
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Assign train/val/test splits (patient-level to prevent leakage)
    rng = np.random.RandomState(seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)
    n = len(patients)
    train_pats = set(patients[: int(0.7 * n)])
    val_pats = set(patients[int(0.7 * n) : int(0.85 * n)])

    df["split"] = df["patient_id"].apply(
        lambda pid: "train" if pid in train_pats else ("val" if pid in val_pats else "test")
    )

    df.to_csv(output, index=False)
    print(f"\nManifest saved to {output}")
    print(f"  Total: {len(df)} images, {df.patient_id.nunique()} patients")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {(df.split == s).sum()} images")
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare NIH ChestX-ray14 manifest")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing extracted X-ray images")
    parser.add_argument("--max_images", type=int, default=5000,
                        help="Maximum number of images to include (0 = all)")
    parser.add_argument("--output", type=str, default="manifest.csv",
                        help="Output manifest CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_manifest(
        data_dir=args.data_dir,
        max_images=args.max_images if args.max_images > 0 else None,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
