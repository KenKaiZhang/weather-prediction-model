import torch
import torch.nn as nn

from deeplearning.base_model import BaseModel


class WeatherCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv1d(
            in_channels=self.input_size, out_channels=64, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 14, 7 * self.input_size)

        self.setup_training()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), 7, self.input_size)
