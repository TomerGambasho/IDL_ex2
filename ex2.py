# === Imports ===
import numpy as np
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils import data as data_utils
from Constants import *
from Classes import Classifier, Autoencoder
from functions_and_utils import *

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 16
base_channels = 4

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform)
mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True,
                                    transform=transform)

# === Subsample for the 100-sample case ===
subset_indices = torch.randperm(len(mnist_dataset))[:NUM_SAMPLES]
subset_dataset = data_utils.Subset(mnist_dataset, subset_indices)

# Full dataset: standard train/test split
train_loader_full = full_dataset_loader(mnist_dataset)

train_loader_100 = dataset_to_loader(subset_dataset)

test_loader_common = test_dataset_loader(mnist_dataset_test)  # Common test set
test_loader_common = data_utils.DataLoader(
    test_loader_common.dataset,
    batch_size=10,
    sampler=torch.utils.data.RandomSampler(test_loader_common.dataset)
)

# === part 1 ===
autoencoder = Autoencoder(latent_dim, base_channels)
train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_autoencoder(
    autoencoder, train_loader_full, test_loader_common)
final_acc_full = acc_test_full[-1]

plt.figure(figsize=(8, 5))
plt.plot(train_loss_full, label="Train Full")
plt.plot(test_loss_full, label="Test Full")
plt.xticks(range(len(train_loss_full)))
plt.title(f"Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === part 2 ===
model_full = Classifier(latent_dim, base_channels, False)
model_100 = Classifier(latent_dim, base_channels, False)

train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier_recognize_the_num(
    model_full, train_loader_full, test_loader_common, name="Full",
    visualize=True)

train_loss_100, test_loss_100, acc_train_100, acc_test_100 = train_classifier(
    model_100, train_loader_100, test_loader_common, name="Subset",
    visualize=True)

# === Final Accuracies ===
final_acc_full = acc_test_full[-1]
final_acc_100 = acc_test_100[-1]

# === Plot Losses with Accuracy in the Title ===
plt.figure(figsize=(8, 5))
plt.plot(train_loss_full, label="Train Full")
plt.plot(test_loss_full, label="Test Full")
plt.plot(train_loss_100, label="Train 100")
plt.plot(test_loss_100, label="Test 100")
plt.title(
    f"Loss over Epochs\nFinal Test Acc - Full: {final_acc_full:.2%}, Subset: {final_acc_100:.2%}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === part 3 ===
model_full = Classifier(latent_dim, base_channels, True)
model_100 = Classifier(latent_dim, base_channels, True)

train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier_recognize_the_num(
    model_full, train_loader_full, test_loader_common, name="Full",
    visualize=True)

train_loss_100, test_loss_100, acc_train_100, acc_test_100 = train_classifier(
    model_100, train_loader_100, test_loader_common, name="Subset",
    visualize=True)

# === Final Accuracies ===
final_acc_full = acc_test_full[-1]
final_acc_100 = acc_test_100[-1]

# === Plot Losses with Accuracy in the Title ===
plt.figure(figsize=(8, 5))
plt.plot(train_loss_full, label="Train Full")
plt.plot(test_loss_full, label="Test Full")
plt.plot(train_loss_100, label="Train 100")
plt.plot(test_loss_100, label="Test 100")
plt.title(
    f"Loss over Epochs\nFinal Test Acc - Full: {final_acc_full:.2%}, Subset: {final_acc_100:.2%}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === part 4 ===
model_full = Classifier(latent_dim, base_channels)
model_100 = Classifier(latent_dim, base_channels)

train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier(
    model_full, train_loader_full, test_loader_common, name="Full")

# === Final Accuracies ===
# final_acc_full = acc_test_full[-1]
# final_acc_100 = acc_test_100[-1]


# === Plot Losses with Accuracy in the Title ===
plt.figure(figsize=(8, 5))
plt.plot(train_loss_full, label="Train Full")
plt.plot(test_loss_full, label="Test Full")
plt.title(
    f"Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
