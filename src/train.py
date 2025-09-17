import torch, os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.dataset import DeepfakeDataset
from src.model import ResNextLSTM
from src import config

def train_model():
   # Example: binary classification (real=0, fake=1)
   real_data = DeepfakeDataset("data/processed/sample_video/frames", label=0)
   fake_data = DeepfakeDataset("data/processed/sample_video/frames", label=1)  # duplicate for demo
   dataset = real_data + fake_data   # concat
   loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

   model = ResNextLSTM().to(config.DEVICE)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=config.LR)

   print("Training started on", config.DEVICE)
   for epoch in range(config.EPOCHS):
      total_loss = 0
      for imgs, labels in loader:
         imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
         # Fake sequence dimension: (B, seq_len=1, C,H,W)
         imgs = imgs.unsqueeze(1)

         optimizer.zero_grad()
         outputs = model(imgs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         total_loss += loss.item()
      print(f"Epoch {epoch+1}/{config.EPOCHS} Loss: {total_loss:.4f}")

   os.makedirs(config.MODEL_DIR, exist_ok=True)
   torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "deepfake_model.pth"))
   print("Model saved at models/deepfake_model.pth")

if __name__ == "__main__":
   train_model()
