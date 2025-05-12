# === Imports ===
import os
import numpy as np
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# === Subsample for the 100-sample case ===
subset_indices = torch.randperm(len(mnist_dataset))[:NUM_SAMPLES]
subset_dataset = data_utils.Subset(mnist_dataset, subset_indices)

# === Train/Test Split Functions ===
def split_dataset(dataset):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    def to_tensor_loader(X, y):
        tensor_X = torch.tensor(np.stack(X)).float()
        tensor_y = torch.tensor(y).long()
        ds = data_utils.TensorDataset(tensor_X, tensor_y)
        return data_utils.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    return to_tensor_loader(X_train, y_train), to_tensor_loader(X_test, y_test)

# Full dataset: standard train/test split
train_loader_full, test_loader_full = split_dataset(mnist_dataset)

# Use all 100 samples for training
def dataset_to_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=True)

train_loader_100 = dataset_to_loader(subset_dataset)

# Use part of the full dataset as a test set for evaluating both models
_, test_loader_100 = split_dataset(mnist_dataset)

# === AutoEncoder Components ===
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
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
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=KERNEL_SIZE,
                               stride=STRIDE, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=KERNEL_SIZE,
                               stride=STRIDE, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.model(x)
        x = self.deconv(x)
        return x

class Classifier(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.encoder = Encoder(latent_dim, base_channels)
        self.classifier = nn.Linear(latent_dim, 10)
    def forward(self, x):
        z = self.encoder(x)
        out = self.classifier(z)
        return out

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 16
base_channels = 4

# === Training function ===
def train_classifier(model, train_loader, test_loader, name=""):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.to(device)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / total
        test_accuracy = correct / total
        test_losses.append(avg_test_loss)
        test_accs.append(test_accuracy)

        print(f"{name} Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}")

    return train_losses, test_losses, train_accs, test_accs

# === Train models ===
model_full = Classifier(latent_dim, base_channels)
model_100 = Classifier(latent_dim, base_channels)

train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier(
    model_full, train_loader_full, test_loader_full, name="Full")

train_loss_100, test_loss_100, acc_train_100, acc_test_100 = train_classifier(
    model_100, train_loader_100, test_loader_100, name="Subset")

# === Final Accuracies ===
final_acc_full = acc_test_full[-1]
final_acc_100 = acc_test_100[-1]

# === Plot Losses with Accuracy in the Title ===
plt.figure(figsize=(8, 5))
plt.plot(train_loss_full, label="Train Full")
plt.plot(test_loss_full, label="Test Full")
plt.plot(train_loss_100, label="Train 100")
plt.plot(test_loss_100, label="Test 100")
plt.title(f"Loss over Epochs\nFinal Test Acc - Full: {final_acc_full:.2%}, Subset: {final_acc_100:.2%}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# # === Visualize Sample Predictions ===
# model_full.eval()
# test_images, test_labels = next(iter(loader_full))  # Get a batch
# test_images = test_images.to(device)
#
# # Run inference
# with torch.no_grad():
#     logits = model_full(test_images)
#     predictions = torch.argmax(logits, dim=1)
#
# # Move data to CPU for visualization
# test_images = test_images.cpu().numpy()
# predictions = predictions.cpu().numpy()
# test_labels = test_labels.numpy()
#
# # Show N sample predictions
# n = 20
# plt.figure(figsize=(15, 3))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.imshow(test_images[i].squeeze(), cmap="gray")
#     pred_label = predictions[i]
#     true_label = test_labels[i]
#     ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}",
#                  color=("green" if pred_label == true_label else "red"))
#     ax.axis("off")
# plt.tight_layout()
# plt.show()
#
#
