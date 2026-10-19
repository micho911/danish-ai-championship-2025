#!/usr/bin/env python3
"""
Create an nnU-Net v2 2D dataset from:
data/
  controls/
    imgs/
    labels/   (may be empty or already zero)
    masks/    (unused here)
  patients/
    imgs/
    labels/   (manual segmentations)

Output:
nnUNet_raw/Dataset<id>_<name>/
  imagesTr/ case_XXXX_0000.png
  labelsTr/ case_XXXX.png
  dataset.json

Patient labels are binarized (>0 -> 1). Controls use zero labels.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------- I/O helpers ----------
def read_gray(path: Path) -> np.ndarray:
    """Read image as grayscale uint8 (robust to RGB/RGBA inputs)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    if img.ndim == 3:
        # Drop alpha if present, convert to gray
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ensure uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def write_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")


# ---------- pairing helpers ----------
_num_pat = re.compile(r"(\d+)")


def extract_number(stem: str) -> str:
    m = _num_pat.search(stem)
    return m.group(1) if m else ""


def build_number_map(paths: List[Path]) -> Dict[str, Path]:
    out = {}
    for p in sorted(paths):
        num = extract_number(p.stem)
        if not num:
            continue
        if num in out:
            # Keep first; warn
            print(f"[warn] Duplicate number {num}: keeping {out[num].name}, ignoring {p.name}")
            continue
        out[num] = p
    return out


# ---------- main conversion ----------
def convert(
    data_root: Path,
    out_root: Path,
    dataset_id: int,
    dataset_name: str,
    use_controls: bool = True,
) -> None:
    controls_img_dir = data_root / "controls" / "imgs"
    controls_lbl_dir = data_root / "controls" / "labels"  # expected (zeros)
    patients_img_dir = data_root / "patients" / "imgs"
    patients_lbl_dir = data_root / "patients" / "labels"

    task_folder = out_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    imagesTr = task_folder / "imagesTr"
    labelsTr = task_folder / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    # Pair patients by shared number (e.g., patient_012 with segmentation_012)
    pat_imgs = sorted(patients_img_dir.glob("*"))
    pat_lbls = sorted(patients_lbl_dir.glob("*"))
    map_img = build_number_map(pat_imgs)
    map_lbl = build_number_map(pat_lbls)

    missing = sorted(set(map_img.keys()) - set(map_lbl.keys()))
    if missing:
        ex = ", ".join(sorted(missing)[:8])
        raise FileNotFoundError(
            f"Missing patient label(s) for numbers: {ex} (+{max(0, len(missing)-8)} more). "
            f"Make sure filenames share a numeric id, e.g., patient_012.* <-> segmentation_012.*"
        )

    # Controls: labels may exist; if not, we will create zeros
    ctl_imgs = sorted((controls_img_dir.glob("*"))) if use_controls else []
    ctl_lbls = sorted((controls_lbl_dir.glob("*"))) if use_controls else []
    map_ctl_img = build_number_map(ctl_imgs)
    map_ctl_lbl = build_number_map(ctl_lbls)  # may be empty

    training_entries = []
    case_idx = 0

    # --- patients ---
    empty_patient_labels = 0
    for num in sorted(map_img.keys()):
        if num not in map_lbl:
            continue  # already checked; just safe-guard

        img_path = map_img[num]
        lbl_path = map_lbl[num]

        # read data
        img = read_gray(img_path)
        lbl = read_gray(lbl_path)

        # binarize: >0 -> 1
        bin_lbl = (lbl > 0).astype(np.uint8)

        if bin_lbl.sum() == 0:
            empty_patient_labels += 1
            print(f"[warn] Patient label appears empty (all-zero): {lbl_path.name}")

        case_id = f"case_{case_idx:04d}"
        write_png(imagesTr / f"{case_id}_0000.png", img)
        write_png(labelsTr / f"{case_id}.png", bin_lbl)

        training_entries.append(
            {"image": f"./imagesTr/{case_id}_0000.png", "label": f"./labelsTr/{case_id}.png"}
        )
        case_idx += 1

    # --- controls ---
    created_zero_controls = 0
    if use_controls:
        for num in sorted(map_ctl_img.keys()):
            img_path = map_ctl_img[num]
            img = read_gray(img_path)

            # choose existing label if present, else create zero
            if num in map_ctl_lbl:
                lbl = read_gray(map_ctl_lbl[num])
                # enforce zero mask for controls
                if np.any(lbl > 0):
                    print(f"[warn] Control label not empty: forcing zero mask for {map_ctl_lbl[num].name}")
                    lbl = np.zeros_like(img, dtype=np.uint8)
            else:
                lbl = np.zeros_like(img, dtype=np.uint8)
                created_zero_controls += 1

            # binarize again to 0/1
            bin_lbl = (lbl > 0).astype(np.uint8)

            case_id = f"case_{case_idx:04d}"
            write_png(imagesTr / f"{case_id}_0000.png", img)
            write_png(labelsTr / f"{case_id}.png", bin_lbl)
            training_entries.append(
                {"image": f"./imagesTr/{case_id}_0000.png", "label": f"./labelsTr/{case_id}.png"}
            )
            case_idx += 1

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {"0": "MIP-PET"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(training_entries),
        "file_ending": ".png",
        "training": training_entries,
        "test": [],
        # Optional metadata nnU-Net v2 reads but does not require:
        # "name": dataset_name, "description": "", "reference": "", "licence": "", "release": "1.0",
    }
    with open(task_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # --- summary ---
    print("\n=== Summary ===")
    print(f"Patients paired: {len(map_img)}")
    print(f"Controls used:   {len(map_ctl_img) if use_controls else 0}")
    print(f"Empty patient labels detected: {empty_patient_labels}")
    if use_controls:
        print(f"Control zero labels created:   {created_zero_controls}")
    print(f"Total training cases: {len(training_entries)}")
    print(f"nnU-Net dataset root: {task_folder.resolve()}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Build nnU-Net v2 2D dataset from PET MIP data.")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Path to data/ containing controls/ and patients/")
    parser.add_argument("--out_root", type=Path,
                        default=Path(os.environ.get("nnUNet_raw", "nnUNet_raw")),
                        help="Where to create DatasetXXX_NAME (defaults to $nnUNet_raw or ./nnUNet_raw)")
    parser.add_argument("--dataset_id", type=int, default=123, help="Three-digit dataset id (e.g., 123)")
    parser.add_argument("--dataset_name", type=str, default="MyTumor", help="Dataset name (no spaces)")
    parser.add_argument("--no_controls", action="store_true", help="Exclude controls from training set")
    args = parser.parse_args()

    convert(
        data_root=args.data_root,
        out_root=args.out_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        use_controls=not args.no_controls,
    )


if __name__ == "__main__":
    main()