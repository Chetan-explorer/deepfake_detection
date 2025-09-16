import torch.nn as nn
import torchvision.models as models
from src import config

class ResNextLSTM(nn.Module):
   def __init__(self, hidden_dim=256, num_classes=config.NUM_CLASSES):
      super().__init__()
      base = models.resnext50_32x4d(pretrained=True)
      self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
      self.lstm = nn.LSTM(2048, hidden_dim, batch_first=True)
      self.fc = nn.Linear(hidden_dim, num_classes)

   def forward(self, x):
      # x: (batch, seq_len, C, H, W)
      b, s, c, h, w = x.size()
      feats = []
      for i in range(s):
         f = self.feature_extractor(x[:, i])
         feats.append(f.view(b, -1))
      feats = torch.stack(feats, dim=1)
      _, (hn, _) = self.lstm(feats)
      out = self.fc(hn[-1])
      return out
