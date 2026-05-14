"""
config.py
=========
Every setting for the Fraud Detection Engine lives here, so the rest of the
code never hard-codes a path, a number, or a label mapping. Change something
once, here, and the whole pipeline follows.

Used by: data.py, models.py, evaluate.py, run_pipeline.py and the notebook.
"""

from pathlib import Path

# --- Folders --------------------------------------------------------------
# Paths are resolved relative to this file, so the project runs the same on
# every team member's machine (no absolute paths).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ExamCheatingDataset"
TRAIN_DIR = DATA_DIR / "train"          # the labelled images
RESULTS_DIR = PROJECT_ROOT / "results"  # comparison table (CSV)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"  # saved plots

# --- Reproducibility ------------------------------------------------------
# One fixed seed used everywhere (split, SMOTE, models) so results never move.
RANDOM_SEED = 42

# --- Feature extraction ---------------------------------------------------
# Each image is resized to this fixed size and turned into a flat vector.
# 32 x 32 grayscale -> 1024 features. Small = fast to train and easy to explain.
IMAGE_SIZE = (32, 32)

# --- Train / test split ---------------------------------------------------
TEST_SIZE = 0.20  # 80% train, 20% test

# --- Labels ---------------------------------------------------------------
# The dataset has 5 raw folders. We collapse them to a BINARY target:
# "normal act" -> 0 (Normal); every other category -> 1 (Fraud).
CATEGORY_TO_BINARY = {
    "normal act": 0,
    "looking friend": 1,
    "giving code": 1,
    "giving object": 1,
    "cheating": 1,
}
LABEL_NAMES = {0: "Normal", 1: "Fraud"}

# Image file types to accept (the dataset mixes .jpg / .JPG / .png / .PNG).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}