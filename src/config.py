import torch
import os

# Paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "models"
EVAL_DIR = os.path.join(MODEL_DIR, "eval")

# Frame extraction
FRAME_RATE = 5   # take every nth frame

# Face detection
IMG_SIZE = 224
MARGIN = 20

# Training
BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
