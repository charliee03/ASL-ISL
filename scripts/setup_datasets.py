#!/usr/bin/env python3
import argparse
import os
import zipfile
from pathlib import Path


def setup_wlasl(data_dir):
    wlasl_dir = Path(data_dir) / "wlasl"
    wlasl_dir.mkdir(parents=True, exist_ok=True)
    print(f"WLASL data directory: {wlasl_dir}")
    print()
    print("To download WLASL:")
    print("  1. Visit: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
    print("  2. Download the dataset zip")
    print(f"  3. Place the zip in {wlasl_dir}/")
    print("  4. Run: python scripts/setup_datasets.py --extract-wlasl")
    print()


def setup_isl(data_dir):
    isl_dir = Path(data_dir) / "isl"
    isl_dir.mkdir(parents=True, exist_ok=True)
    print(f"ISL data directory: {isl_dir}")
    print()
    print("To download ISL-CSLGR:")
    print("  1. Visit: https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset")
    print("  2. Download the dataset zip")
    print(f"  3. Place the zip in {isl_dir}/")
    print("  4. Run: python scripts/setup_datasets.py --extract-isl")
    print()


def extract_wlasl(data_dir):
    wlasl_dir = Path(data_dir) / "wlasl"
    zip_path = wlasl_dir / "wlasl-processed.zip"
    if not zip_path.exists():
        print(f"Zip not found at {zip_path}. Download first.")
        return
    print("Extracting WLASL...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(wlasl_dir)
    print("Done.")


def extract_isl(data_dir):
    isl_dir = Path(data_dir) / "isl"
    zip_path = isl_dir / "isl-csltr.zip"
    if not zip_path.exists():
        print(f"Zip not found at {zip_path}. Download first.")
        return
    print("Extracting ISL-CSLGR...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(isl_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup AITE datasets")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--extract-wlasl", action="store_true", help="Extract WLASL zip")
    parser.add_argument("--extract-isl", action="store_true", help="Extract ISL zip")
    args = parser.parse_args()
    if not (args.extract_wlasl or args.extract_isl):
        setup_wlasl(args.data_dir)
        setup_isl(args.data_dir)
    else:
        if args.extract_wlasl:
            extract_wlasl(args.data_dir)
        if args.extract_isl:
            extract_isl(args.data_dir)
