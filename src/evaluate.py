import torch
from torch.utils.data import DataLoader
from src.dataset import DeepfakeDataset
from src.model import ResNextLSTM
from src import config

def evaluate_model():
   dataset = DeepfakeDataset("data/processed/sample_video/frames", label=1)
   loader = DataLoader(dataset, batch_size=2)

   model = ResNextLSTM().to(config.DEVICE)
   model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=config.DEVICE))
   model.eval()

   correct, total = 0, 0
   with torch.no_grad():
      for imgs, labels in loader:
         imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
         imgs = imgs.unsqueeze(1)
         outputs = model(imgs)
         preds = torch.argmax(outputs, dim=1)
         correct += (preds == labels).sum().item()
         total += labels.size(0)

   acc = correct / total if total > 0 else 0
   print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
   evaluate_model()
