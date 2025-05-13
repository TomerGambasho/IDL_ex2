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
base_channels = 16

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

plt.figure(figsize=(8, 5))
plt.plot(train_loss_full, label="Train Full")
plt.plot(test_loss_full, label="Test Full")
plt.title(f"Loss over Epochs\nFinal Test Acc - Full: {acc_test_full:.2%}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === part 2 ===
def train_classifier(model, train_loader, test_loader, name="", visualize=False, n_vis=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='sum')
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

        print(f"{name} Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}")

        # === Visualize Predictions ===
        if visualize:
            vis_images, vis_labels = next(iter(test_loader))  # Random batch
            vis_images = vis_images.to(device)
            vis_labels = vis_labels.to(device)

            with torch.no_grad():
                outputs = model(vis_images)
                preds = torch.argmax(outputs, dim=1)

            plt.figure(figsize=(n_vis * 2, 2.5))
            for i in range(n_vis):
                ax = plt.subplot(1, n_vis, i + 1)
                img = vis_images[i].cpu().squeeze().numpy()
                pred = preds[i].item()
                true = vis_labels[i].item()
                plt.imshow(img, cmap="gray")
                ax.set_title(f"Pred: {pred}\nTrue: {true}",
                             color="green" if pred == true else "red")
                ax.axis("off")
            plt.suptitle(f"{name} Predictions - Epoch {epoch + 1}")
            plt.tight_layout()
            plt.show()


    return train_losses, test_losses, train_accs, test_accs


# === Train models ===
model_full = Classifier(latent_dim, base_channels)
model_100 = Classifier(latent_dim, base_channels)

train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier(
    model_full, train_loader_full, test_loader_common, name="Full", visualize=True)

train_loss_100, test_loss_100, acc_train_100, acc_test_100 = train_classifier(
    model_100, train_loader_100, test_loader_common, name="Subset", visualize=True)

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



# # === Train models ===
# model_full = Classifier(latent_dim, base_channels)
# model_100 = Classifier(latent_dim, base_channels)
#
# train_loss_full, test_loss_full, acc_train_full, acc_test_full = train_classifier(
#     model_full, train_loader_full, test_loader_common, name="Full")
#
# train_loss_100, test_loss_100, acc_train_100, acc_test_100 = train_classifier(
#     model_100, train_loader_100, test_loader_common, name="Subset")
#
# # === Final Accuracies ===
# final_acc_full = acc_test_full[-1]
# final_acc_100 = acc_test_100[-1]
