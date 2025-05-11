# === Imports ===
import os
import numpy as np
import torch
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

# Define the transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
train_loader = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a subset of the dataset for testing
indices = torch.arange(100)
train_loader_CLS = data_utils.Subset(train_loader, indices)

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(train_loader_CLS, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# # Create a figure to display the images
# plt.figure(figsize=(15, 3))
#
# # Print the first few images in a row
# for i, (image, label) in enumerate(train_loader_pytorch):
#     if i < 5:  # Print the first 5 samples
#         plt.subplot(1, 5, i + 1)
#         plt.imshow(image[0].squeeze(), cmap='gray')
#         plt.title(f"Label: {label.item()}")
#         plt.axis('off')
#     else:
#         break  # Exit the loop after printing 5 samples
#
# plt.tight_layout()
# plt.show()
