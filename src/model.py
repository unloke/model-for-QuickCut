# src/model.py
import torch
import torch.nn as nn
from . import config

class StutterDetector(nn.Module):
    def __init__(self):
        super(StutterDetector, self).__init__()
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), 

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), 

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)) 
        )
        
        cnn_output_dim = 64 * (config.N_MELS // 8) 

        # 2. RNN Sequence Modeling
        self.gru = nn.GRU(
            input_size=cnn_output_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. Classifier Head
        self.fc = nn.Linear(64 * 2, config.CLASSES)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x) 
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, x.size(1), -1)
        x, _ = self.gru(x)
        logits = self.fc(x) 
        return logits