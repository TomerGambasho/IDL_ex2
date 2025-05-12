# === Imports ===
import os
import numpy as np
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets, transforms
from torch.utils import data as data_utils

# === Constants and Mappings ===
BATCH_SIZE = 32
EPOCHS = 10
KERNEL_SIZE = 3
STRIDE = 2

# Load the MNIST dataset with the specified transformation
transform = transforms.Compose([transforms.ToTensor()])
train_loader = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transform)


#
# # Create a subset for the test
# indices = torch.arange(100)
# train_loader_CLS = data_utils.Subset(train_loader, indices)
#
# # Create a DataLoader to load the dataset in batches
# train_loader_pytorch = torch.utils.data.DataLoader(train_loader_CLS,
#                                                    batch_size=BATCH_SIZE,
#                                                    shuffle=True, num_workers=0)


# === AutoEncoder Components ===
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=KERNEL_SIZE, stride=STRIDE,
                      padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Linear((base_channels * 2) * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.linear = nn.Linear(latent_dim, (base_channels * 2) * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (base_channels * 2, 7, 7)),
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=KERNEL_SIZE,
                               stride=STRIDE, padding=1, output_padding=1),
            # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=KERNEL_SIZE,
                               stride=STRIDE,
                               padding=1, output_padding=1),
            # 14x14 -> 28x28
            nn.Sigmoid()  # Normalized the output [0,1]
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


# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 16  # You can change to 4 or other values
base_channels = 4  # Or 16 for larger model

model = Autoencoder(latent_dim=latent_dim, base_channels=base_channels).to(
    device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

# Use full MNIST here instead of subset
full_loader = torch.utils.data.DataLoader(train_loader, batch_size=BATCH_SIZE,
                                          shuffle=True)

# === Training Loop ===
losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch in full_loader:
        images, _ = batch
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(full_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    losses.append(avg_loss)

# === Visualization of reconstructions ===
# Set the model to evaluation mode
model.eval()

# Get a batch of test images
test_images, _ = next(iter(full_loader))
test_images = test_images.to(device)
with torch.no_grad():
    reconstructed = model(test_images)

# Convert tensors to numpy arrays for visualization
test_images = test_images.cpu().numpy()
reconstructed = reconstructed.cpu().numpy()

# Plot original and reconstructed images side-by-side
n = 5  # number of images to show
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    ax.axis("off")
    if i == 0:
        ax.set_title("Original", fontsize=14)

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    ax.axis("off")
    if i == 0:
        ax.set_title("Reconstructed", fontsize=14)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), losses, marker='o')
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
