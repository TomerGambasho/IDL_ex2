# === Imports ===
import torch.nn as nn
from Constants import *


# === AutoEncoder Components ===
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=KERNEL_SIZE, stride=STRIDE,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Linear((base_channels * 2) * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear((base_channels * 2) * 7 * 7,
                      min((base_channels * 2) * 7 * 7, 2 * latent_dim)),
            nn.Linear(2 * latent_dim, latent_dim),
            # nn.Dropout(0.3),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.linear = nn.Linear(latent_dim, (base_channels * 2) * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (base_channels * 2, 7, 7)),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),

            # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),

            # 14x14 -> 28x28
            nn.Sigmoid()  # Normalized output [0,1]
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.deconv(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.encoder = Encoder(latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, base_channels)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class Classifier(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.encoder = Encoder(latent_dim, base_channels)
        self.encoder.trainable = False  # Freeze encoder
        self.mlp = MLP(latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, base_channels)

    def forward(self, x):
        x2 = self.encoder(x)
        x3 = self.mlp(x2)
        out = self.decoder(x3)
        return out
