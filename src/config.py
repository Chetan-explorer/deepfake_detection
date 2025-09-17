import os

# Base directories
BASE_DIR = os.path.abspath("data")
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
FRAMES_DIR = os.path.join(PROCESSED_DIR, "frames")   # store extracted frames
FACES_DIR = os.path.join(PROCESSED_DIR, "faces")     # store cropped faces
MODEL_DIR = os.path.join("models")

# Dataset config
IMG_SIZE = 224
NUM_CLASSES = 2  # Real, Fake
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda"  # or "cpu"











'''
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
'''