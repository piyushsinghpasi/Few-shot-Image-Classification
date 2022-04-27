# !pip install torch torchvision
import torch.nn as nn
class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            # assuming input_shape is 512
            nn.Linear(kwargs["input_shape"], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 36),
            # torch.nn.ReLU(),
            # torch.nn.Linear(36, 18),
            # torch.nn.ReLU(),
            # torch.nn.Linear(18, 9)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, kwargs["input_shape"]),
            nn.ReLU(),
            # torch.nn.Linear(36, 64),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 28 * 28),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec