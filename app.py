import streamlit as st
import os, cv2, torch, torch.nn.functional as F
from torchvision import transforms
from src.model import ResNextLSTM
from src import config
from src.face_detect_crop import extract_faces_from_video

# --- Config ---
UPLOAD_DIR = "data/uploads"
FRAMES_DIR = "data/processed/app_frames"
SEQ_LEN = 8

# Transform (same as dataset.py)
transform = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load model once
@st.cache_resource
def load_model():
   model = ResNextLSTM(num_classes=config.NUM_CLASSES).to(config.DEVICE)
   model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=config.DEVICE))
   model.eval()
   return model

def load_frames(folder, seq_len=SEQ_LEN):
   frame_files = sorted(os.listdir(folder))
   frames = []
   for f in frame_files[:seq_len]:
      img_path = os.path.join(folder, f)
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = transform(img)
      frames.append(img)
   if len(frames) < seq_len:
      return None
   video_tensor = torch.stack(frames, dim=0)  # (seq_len, C,H,W)
   video_tensor = video_tensor.unsqueeze(0)   # (1, seq_len, C,H,W)
   return video_tensor

def predict_video(video_path):
   os.makedirs(FRAMES_DIR, exist_ok=True)
   # Extract faces into FRAMES_DIR
   extract_faces_from_video(video_path, FRAMES_DIR, frame_skip=5)

   # Load frames
   x = load_frames(FRAMES_DIR, SEQ_LEN)
   if x is None:
      return "Not enough faces detected"
   x = x.to(config.DEVICE)

   # Run model
   model = load_model()
   with torch.no_grad():
      out = model(x)
      probs = F.softmax(out, dim=1)
      pred = probs.argmax(dim=1).item()
   return "FAKE" if pred == 1 else "REAL"

# --- Streamlit UI ---
st.title("🎭 Deepfake Detection App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mkv"])
if uploaded_file is not None:
   os.makedirs(UPLOAD_DIR, exist_ok=True)
   video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

   with open(video_path, "wb") as f:
      f.write(uploaded_file.getbuffer())
   st.video(video_path)

   if st.button("Run Deepfake Detection"):
      st.write("🔍 Processing video...")
      result = predict_video(video_path)
      st.success(f"Prediction: {result}")











'''
import streamlit as st
from src.inference import predict_image

st.title("🔍 Deepfake Detection App")
uploaded = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
if uploaded:
   with open("temp.jpg", "wb") as f:
      f.write(uploaded.read())
   st.image("temp.jpg", caption="Uploaded", use_column_width=True)
   result = predict_image("temp.jpg")
   st.success(f"Prediction: {result}")

'''