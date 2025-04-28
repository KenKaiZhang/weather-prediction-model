import torch.nn as nn

from deeplearning.base_model import BaseModel


class WeatherTransformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=23, nhead=4, dim_feedforward=128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(23 * 14, 7 * 23)

        self.setup_training()

    def forward(self, x):
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), 7, 23)
