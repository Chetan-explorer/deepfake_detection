import os, cv2, torch
from torch.utils.data import Dataset
from torchvision import transforms
from src import config

transform = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

class DeepfakeDataset(Dataset):
   def __init__(self, root_dir, label=0, transform=transform):
      self.root_dir = root_dir
      self.transform = transform
      self.frames = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
      self.label = label

   def __len__(self):
      return len(self.frames)

   def __getitem__(self, idx):
      img = cv2.imread(self.frames[idx])
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = self.transform(img)
      return img, torch.tensor(self.label)

if __name__ == "__main__":
   dataset = DeepfakeDataset("data/processed/sample_video/frames", label=1)
   print(len(dataset), "frames loaded")
   x, y = dataset[0]
   print("One sample:", x.shape, y)
