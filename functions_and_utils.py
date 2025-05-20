# === Imports ===
import os

import numpy as np
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils import data as data_utils
from Constants import *
from Classes import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Train/Test Split Functions ===
def full_dataset_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=True)


# Use all 100 samples for training
def dataset_to_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=True)


def test_dataset_loader(dataset, batch_size=BATCH_SIZE):
    X = [data[0].numpy() for data in dataset]
    y = [data[1] for data in dataset]
    tensor_X = torch.tensor(np.stack(X)).float()
    tensor_y = torch.tensor(y).long()
    ds = data_utils.TensorDataset(tensor_X, tensor_y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_classifier_recognize_the_num(model, train_loader, test_loader,
                                       name="", visualize=False, n_vis=10):
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

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / len(train_loader)
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

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accs.append(test_accuracy)

        print(
            f"{name} Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
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


# === Training function ===
def train_classifier(model, train_loader, test_loader, name="",
                     visualize=VISUALIZE, n_vis=N_VIS):
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

        avg_train_loss = train_loss / len(train_loader)
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
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
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
            save_dir = "part_4_results"
            os.makedirs(save_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir,
                                     f"reconstruction{epoch+1}.png"))  # Save the figure
            plt.close()

    return train_losses, test_losses, train_accs, test_accs


def train_autoencoder(model, train_loader, test_loader,
                      visualize=VISUALIZE, n_vis=N_VIS):
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
        epoch_loss = 0
        for batch in train_loader:
            images, _ = batch
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                images, _ = batch
                images = images.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accs.append(test_accuracy)

        # print(f"{name} Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
        #       f"Test Loss: {avg_test_loss:.4f}, Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}")
        print(
            f" Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
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
            save_dir = "IDL_ex2/images"
            os.makedirs(save_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir,
                                     f"reconstruction{epoch}.png"))  # Save the figure

            plt.close()

    return train_losses, test_losses, train_accs, test_accs
