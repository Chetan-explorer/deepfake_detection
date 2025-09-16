# for single Video

import os
import cv2
import torch
from facenet_pytorch import MTCNN

# ------------------- CONFIG -------------------
OUT_DIR = "data/faces"     # folder where cropped faces will be saved
FRAME_SKIP = 2             # process every Nth frame (1 = every frame)
RESIZE_SCALE = 0.5         # downscale factor for faster detection
# ----------------------------------------------

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)


def extract_faces_from_video(video_path, out_dir, frame_skip=FRAME_SKIP, resize_scale=RESIZE_SCALE):
   """
   Extract faces from a single video and save them into out_dir.
   """
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
      print(f"Cannot open video: {video_path}")
      return

   os.makedirs(out_dir, exist_ok=True)
   frame_idx = 0
   saved_faces = 0

   while True:
      ret, frame = cap.read()
      if not ret:
         break

      # Skip frames to save time
      if frame_idx % frame_skip != 0:
         frame_idx += 1
         continue

      # Resize frame for faster detection
      small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
      rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

      # Detect faces
      boxes, _ = mtcnn.detect(rgb_small_frame)

      if boxes is not None:
         for i, box in enumerate(boxes):
               # Scale coordinates back to original frame size
               x1, y1, x2, y2 = (box / resize_scale).astype(int)
               face = frame[y1:y2, x1:x2]
               if face.size == 0:
                  continue

               # Save cropped face
               face_file = os.path.join(
                  out_dir,
                  f"{os.path.basename(video_path).split('.')[0]}_frame{frame_idx}_face{i}.jpg"
               )
               cv2.imwrite(face_file, face)
               saved_faces += 1

      frame_idx += 1
      # Clear GPU memory after each frame
      torch.cuda.empty_cache()

   cap.release()
   print(f"✅ Extracted {saved_faces} faces from {video_path} → {out_dir}")


# ------------------- RUN -------------------
if __name__ == "__main__":
   # Specify your video path here
   video_path = "data/raw/sample.mp4"  # change to your test video

   # Decide if the video is real or fake based on folder name
   category = "real" if "original" in video_path else "fake"
   # output_dir = os.path.join(OUT_DIR, category)
   output_dir = "data/processed/sample_video/frames"

   # Extract faces
   extract_faces_from_video(video_path, output_dir)























'''

import os
import cv2
import torch
from facenet_pytorch import MTCNN

# ------------------- CONFIG -------------------
RAW_DIR = "data/raw"       # original dataset folder (videos remain here)
OUT_DIR = "data/faces"     # where cropped faces will be saved
FRAME_SKIP = 2             # process every Nth frame (1 = every frame)
RESIZE_SCALE = 0.5         # downscale factor for faster detection
# ----------------------------------------------

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)


def extract_faces_from_video(video_path, out_dir, frame_skip=FRAME_SKIP, resize_scale=RESIZE_SCALE):
   """
   Extract faces from a single video and save them into out_dir.
   """
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
      print(f"Cannot open video: {video_path}")
      return

   os.makedirs(out_dir, exist_ok=True)
   frame_idx = 0
   saved_faces = 0

   while True:
      ret, frame = cap.read()
      if not ret:
         break

      # Skip frames to save time
      if frame_idx % frame_skip != 0:
         frame_idx += 1
         continue

      # Resize frame for faster detection
      small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
      rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

      # Detect faces
      boxes, _ = mtcnn.detect(rgb_small_frame)

      if boxes is not None:
         for i, box in enumerate(boxes):
               # Scale coordinates back to original frame size
               x1, y1, x2, y2 = (box / resize_scale).astype(int)
               face = frame[y1:y2, x1:x2]
               if face.size == 0:
                  continue

               # Save cropped face
               face_file = os.path.join(
                  out_dir,
                  f"{os.path.basename(video_path).split('.')[0]}_frame{frame_idx}_face{i}.jpg"
               )
               cv2.imwrite(face_file, face)
               saved_faces += 1

      frame_idx += 1

      # Clear GPU memory after each frame
      torch.cuda.empty_cache()

   cap.release()
   print(f"✅ Extracted {saved_faces} faces from {video_path} → {out_dir}")


# ------------------- RUN -------------------
if __name__ == "__main__":
   # Process both original and manipulated videos
   for category in ["original", "manipulated"]:
      cat_raw_dir = os.path.join(RAW_DIR, category)
      cat_out_dir = os.path.join(OUT_DIR, "real" if category == "original" else "fake")

      # Recursively process all videos
      for root, dirs, files in os.walk(cat_raw_dir):
         for file in files:
               if file.endswith((".mp4", ".avi", ".mkv")):
                  video_path = os.path.join(root, file)
                  # Preserve folder structure inside faces/real or faces/fake
                  video_out_dir = os.path.join(cat_out_dir, os.path.relpath(root, cat_raw_dir))
                  extract_faces_from_video(video_path, video_out_dir)

'''