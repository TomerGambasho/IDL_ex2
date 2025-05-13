# === Imports ===
import numpy as np
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils import data as data_utils

# === Constants and Mappings ===
BATCH_SIZE = 32
EPOCHS = 20
KERNEL_SIZE = 3
STRIDE = 2
NUM_SAMPLES = 100
VISUALIZE = True
N_VIS = 5

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 16
base_channels = 16

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# === Subsample for the 100-sample case ===
subset_indices = torch.randperm(len(mnist_dataset))[:NUM_SAMPLES]
subset_dataset = data_utils.Subset(mnist_dataset, subset_indices)


# === Train/Test Split Functions ===
def full_dataset_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=True)


# Full dataset: standard train/test split
train_loader_full = full_dataset_loader(mnist_dataset)


# Use all 100 samples for training
def dataset_to_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=True)


train_loader_100 = dataset_to_loader(subset_dataset)


def test_dataset_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=False)


test_loader_common = test_dataset_loader(mnist_dataset_test)  # Common test set
test_loader_common = data_utils.DataLoader(
    test_loader_common.dataset,
    batch_size=10,
    sampler=torch.utils.data.RandomSampler(test_loader_common.dataset)
)


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

    def forward(self, x):
        x = self.conv(x)

        return x


class MLP(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear((base_channels * 2) * 7 * 7, min((base_channels * 2) * 7 * 7, 2 * latent_dim)),
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
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),

            # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),

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


# === Training function ===
def train_classifier(model, train_loader, test_loader, name="", visualize=VISUALIZE, n_vis=N_VIS):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    model.to(device)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Preload a batch of test samples for consistent visualization
    if visualize:
        vis_images, vis_labels = next(iter(test_loader))
        vis_images = vis_images.to(device)
        vis_labels = vis_labels.to(device)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # _, predicted = torch.max(outputs, 1)
            # correct += (predicted == images).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        # train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        # train_accs.append(train_accuracy)

        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item()
                # _, predicted = torch.max(outputs, 1)
                # correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / total
        # test_accuracy = correct / total
        test_losses.append(avg_test_loss)
        # test_accs.append(test_accuracy)

        # print(f"{name} Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
        #       f"Test Loss: {avg_test_loss:.4f}, Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}")
        print(f"{name} Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}.")

        # === Visualize Predictions ===
        if visualize:
            vis_images, vis_labels = next(iter(test_loader))  # Random batch
            vis_images = vis_images.to(device)
            vis_labels = vis_labels.to(device)

            with torch.no_grad():
                outputs = model(vis_images)

            plt.figure(figsize=(20, 4))
            for i in range(N_VIS):
                # Original
                ax = plt.subplot(2, N_VIS, i + 1)
                plt.imshow(vis_images[i].squeeze(), cmap='gray')
                ax.axis("off")
                if i == 0:
                    ax.set_title("Original", fontsize=14)

                # Reconstructed
                ax = plt.subplot(2, N_VIS, i + 1 + N_VIS)
                plt.imshow(outputs[i].squeeze(), cmap='gray')
                ax.axis("off")
                if i == 0:
                    ax.set_title("Reconstructed", fontsize=14)

            plt.tight_layout()
            plt.show()

    return train_losses, test_losses, train_accs, test_accs


# === Train models ===
model_full = Classifier(latent_dim, base_channels)
model_100 = Classifier(latent_dim, base_channels)

train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier(
    model_full, train_loader_full, test_loader_common, name="Full")

train_loss_100, test_loss_100, acc_train_100, acc_test_100 = train_classifier(
    model_100, train_loader_100, test_loader_common, name="Subset")


# === Final Accuracies ===
# final_acc_full = acc_test_full[-1]
# final_acc_100 = acc_test_100[-1]
