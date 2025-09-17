import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2

from src.model import ResNextLSTM
from src import config

# ---------------- CONFIG ----------------
VIDEO_FRAMES_DIR = "data/processed/sample_video/frames"  # where your extracted frames are
SEQ_LEN = 8  # how many frames per sequence
# ----------------------------------------

# Transformation (same as dataset.py)
transform = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def load_frames(folder, seq_len=SEQ_LEN):
   """Load frames from a folder, return tensor (1, seq_len, 3, H, W)."""
   frame_files = sorted(os.listdir(folder))
   frames = []

   for f in frame_files[:seq_len]:  # take first N frames
      img_path = os.path.join(folder, f)
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = transform(img)
      frames.append(img)

   if len(frames) < seq_len:
      raise ValueError(f"Not enough frames found, need {seq_len} but got {len(frames)}")

   video_tensor = torch.stack(frames, dim=0)  # (seq_len, C, H, W)
   video_tensor = video_tensor.unsqueeze(0)   # (1, seq_len, C, H, W)
   return video_tensor


if __name__ == "__main__":
   # 1. Load model
   model = ResNextLSTM(num_classes=config.NUM_CLASSES)
   model.eval()  # eval mode

   # 2. Load frames from video
   x = load_frames(VIDEO_FRAMES_DIR, SEQ_LEN)

   # 3. Run inference
   with torch.no_grad():
      out = model(x)
      probs = F.softmax(out, dim=1)
      pred = torch.argmax(probs, dim=1).item()

   # 4. Map prediction to label
   labels = ["REAL", "FAKE"]
   print(f"Prediction: {labels[pred]} (confidence: {probs[0][pred].item():.4f})")
