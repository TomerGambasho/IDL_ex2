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
EPOCHS = 20
KERNEL_SIZE = 3
STRIDE = 2
NUM_SAMPLES = 100

# Load the MNIST dataset with the specified transformation
transform = transforms.Compose([transforms.ToTensor()])
train_loader = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transform)

# === Added for NUM_SAMPLES-sample training ===
# Select NUM_SAMPLES random indices from the full dataset
subset_indices = torch.randperm(len(train_loader))[:NUM_SAMPLES]
train_loader_100 = data_utils.Subset(train_loader, subset_indices)

# Create DataLoaders for both full and subset
loader_full = torch.utils.data.DataLoader(train_loader, batch_size=BATCH_SIZE,
                                          shuffle=True)
loader_100 = torch.utils.data.DataLoader(train_loader_100, batch_size=BATCH_SIZE,
                                         shuffle=True)

# === Preprocessing Functions ===
def encode_label(label):
    one_hot = np.zeros((1, 10), dtype=np.float32)
    one_hot[0, label] = 1.0
    return torch.tensor(one_hot.flatten())

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
        self.model = nn.Sequential(
            nn.Linear((base_channels * 2) * 7 * 7, latent_dim),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, (base_channels * 2) * 7 * 7),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
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
        x = self.model(x)
        x = self.deconv(x)
        return x


class Classifier(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.encoder = Encoder(latent_dim, base_channels)
        self.classifier = nn.Linear(latent_dim, 10)  # MLP layer

    def forward(self, x):
        z = self.encoder(x)
        out = self.classifier(z)
        return out


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

# === Training function to support both loaders ===
def train_classifier(model, loader, name=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='sum')  # Sum the loss over the batch
    model = model.to(device)

    losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item()
            total_samples += batch_size

        avg_loss = running_loss / total_samples  # Normalize by total number of samples
        print(f"{name} Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

    return losses


# === Added: Train two models ===
model_full = Classifier(latent_dim, base_channels)
model_100 = Classifier(latent_dim, base_channels)

losses_full = train_classifier(model_full, loader_full, name="Full")
losses_100 = train_classifier(model_100, loader_100, name=f"{NUM_SAMPLES}-sample")

# === Plot both losses ===
plt.figure(figsize=(8, 5))
plt.plot(losses_full, label="Full dataset")
plt.plot(losses_100, label=f"{NUM_SAMPLES} examples")
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Visualize Sample Predictions ===
model_full.eval()
test_images, test_labels = next(iter(loader_full))  # Get a batch
test_images = test_images.to(device)

# Run inference
with torch.no_grad():
    logits = model_full(test_images)
    predictions = torch.argmax(logits, dim=1)

# Move data to CPU for visualization
test_images = test_images.cpu().numpy()
predictions = predictions.cpu().numpy()
test_labels = test_labels.numpy()

# Show N sample predictions
n = 20
plt.figure(figsize=(15, 3))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    pred_label = predictions[i]
    true_label = test_labels[i]
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}",
                 color=("green" if pred_label == true_label else "red"))
    ax.axis("off")
plt.tight_layout()
plt.show()
