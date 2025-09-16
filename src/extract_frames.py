import cv2, os
from src import config

# allowed video formats
ALLOWED_EXTS = {".mp4", ".avi", ".mkv"}

def extract_frames(video_path, out_dir, every_n=config.FRAME_RATE):
   # check file exists
   if not os.path.exists(video_path):
      raise FileNotFoundError(f"Video not found: {video_path}")

   # check extension
   _, ext = os.path.splitext(video_path)
   if ext.lower() not in ALLOWED_EXTS:
      raise ValueError(f"Unsupported video format: {ext}. Allowed: {ALLOWED_EXTS}")

   os.makedirs(out_dir, exist_ok=True)
   cap = cv2.VideoCapture(video_path)

   if not cap.isOpened():
      raise ValueError(f"Could not open video: {video_path}")

   idx, saved = 0, 0
   while True:
      ret, frame = cap.read()
      if not ret:
         break
      if idx % every_n == 0:
         save_path = os.path.join(out_dir, f"frame_{idx}.jpg")
         cv2.imwrite(save_path, frame)
         saved += 1
      idx += 1

   cap.release()
   print(f"✅ Extracted {saved} frames → {out_dir}")

if __name__ == "__main__":
   test_video = os.path.join(config.RAW_DIR, "sample.mp4")
   out_dir = os.path.join(config.RAW_DIR, "sample_video", "frames")
   extract_frames(test_video, out_dir)
