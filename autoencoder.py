import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super(AE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
        )
        # Bottleneck
        self.bottleneck = nn.Linear(hidden_dim, hidden_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.bottleneck(h)
        out = self.decoder(z)
        return out