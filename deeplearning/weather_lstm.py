import torch.nn as nn

from deeplearning.base_model import BaseModel


class WeatherLSTM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(256, 7 * self.input_size)
        self.dropout = nn.Dropout(0.3)

        self.setup_training()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        x = self.fc(x)
        return x.view(x.size(0), 7, self.input_size)
