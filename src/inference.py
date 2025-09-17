import os
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms

from src.model import ResNextLSTM
from src import config

# ---------------- CONFIG ----------------
VIDEO_FRAMES_DIR = "data/processed/sample_video/frames"
SEQ_LEN = 8   # number of frames per sequence
# ----------------------------------------

# Same preprocessing as dataset.py
transform = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def load_frames(folder, seq_len=SEQ_LEN):
   """Load a fixed number of frames from folder."""
   frame_files = sorted(os.listdir(folder))
   frames = []

   for f in frame_files[:seq_len]:  # take first N frames
      img_path = os.path.join(folder, f)
      img = cv2.imread(img_path)
      if img is None:
         raise FileNotFoundError(f"Could not read {img_path}")
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = transform(img)
      frames.append(img)

   if len(frames) < seq_len:
      raise ValueError(f"Not enough frames: need {seq_len}, found {len(frames)}")

   video_tensor = torch.stack(frames, dim=0)   # (seq_len, C, H, W)
   video_tensor = video_tensor.unsqueeze(0)    # (1, seq_len, C, H, W)
   return video_tensor

def predict_video(folder):
   model = ResNextLSTM(num_classes=config.NUM_CLASSES).to(config.DEVICE)
   model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=config.DEVICE))
   model.eval()

   x = load_frames(folder, SEQ_LEN).to(config.DEVICE)

   with torch.no_grad():
      out = model(x)
      probs = F.softmax(out, dim=1)
      pred = probs.argmax(dim=1).item()

   return pred, probs.cpu().numpy()

if __name__ == "__main__":
   pred, probs = predict_video(VIDEO_FRAMES_DIR)
   label = "REAL" if pred == 0 else "FAKE"
   print("Prediction:", label)
   print("Probabilities:", probs)
