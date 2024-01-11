import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils.visualization import print_progress_bar
from utils.constants import DEVICE, EPOCHS, BATCH_SIZE, N_WORKERS


def plot_learning_curve(losses: dict, num_epochs: int = 100, size: tuple = (8, 5), meta: dict = {
    'train': {
        'title': 'Training Loss',
        'color': '#ff5a7d'
    },
    'validation': {
        'title': 'Validation Loss',
        'color': '#ff9e00'
    }
}) -> None:
    """
    Plots the learning curve for training and validation losses.

    Args:
    - losses (dict): Dictionary containing training and validation losses.
    - num_epochs (int): Number of epochs to plot.
    - size (tuple): Size of the plot (default: (8, 5)).
    - meta (dict): Metadata for plotting including title and color for each loss group.

    Returns:
    - None
    """
    plt.figure(figsize=size)
    num_epochs = min(len(losses['train']), num_epochs)
    for loss_group in losses:
        plt.plot(range(1, num_epochs + 1), losses[loss_group], marker='o', linestyle='-',
                 color=meta[loss_group]['color'], label=meta[loss_group]['title'])

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()


def calc_validation_perform(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, device: str = DEVICE) -> float:
    """
    Calculates the average validation loss for a given model.

    Args:
    - model (nn.Module): PyTorch model.
    - criterion (nn.Module): Loss function.
    - val_loader (DataLoader): Validation data loader.
    - device (str): Device for computation (default: DEVICE).

    Returns:
    - float: Average validation loss.
    """
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            batch_loss = criterion(outputs, labels).item()
            total_loss += batch_loss

    return total_loss / len(val_loader)


def load_model(model: nn.Module, model_path: str, device: str = DEVICE) -> nn.Module:
    """
    Loads pre-trained model weights from a specified path.

    Args:
    - model (nn.Module): PyTorch model.
    - model_path (str): Path to the saved model weights.
    - device (str): Device for computation (default: DEVICE).

    Returns:
    - nn.Module: Model with loaded weights.
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    return model


def optimize(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader = None, epochs=EPOCHS, device: str = DEVICE, save_path='./model.pt'):
    """
    Trains a PyTorch model using the specified optimizer and loss function.

    Args:
    - model (nn.Module): PyTorch model.
    - criterion (nn.Module): Loss function.
    - optimizer (optim.Optimizer): Optimizer.
    - train_loader (DataLoader): Training data loader.
    - val_loader (DataLoader): Validation data loader (default: None).
    - epochs (int): Number of training epochs (default: EPOCHS).
    - device (str): Device for computation (default: DEVICE).
    - save_path (str): Path to save the trained model (default: './model.pt').

    Returns:
    - dict: Dictionary containing training and validation losses.
    """
    model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        print(f'Epoch: {epoch}/{epochs}')

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f'\r{i+1}/{len(train_loader)} {print_progress_bar(i, len(train_loader))} - Loss: {loss:.6f}', end='', flush=True)

        train_losses.append(total_loss / len(train_loader))
        print(f' | Train Loss: {train_losses[-1]:.4f}', end='')
        if val_loader:
            val_losses.append(calc_validation_perform(
                model, criterion, val_loader))
            print(f' - Val Loss: {val_losses[-1]:.4f}')
        else:
            print()

    torch.save(model.state_dict(), save_path)

    print(f'Completed! =)\nCheck your model saved at {save_path}')

    return {'train': train_losses, 'validation': val_losses}


def evaluate(model: nn.Module, dataloader: DataLoader, predict_only: bool = True, device: str = DEVICE) -> tuple:
    """
    Evaluates a PyTorch model on a given dataset.

    Args:
    - model (nn.Module): PyTorch model.
    - dataloader (DataLoader): Data loader for evaluation.
    - predict_only (bool): If True, returns predictions only; if False, returns predictions, true labels, and probabilities (default: True).
    - device (str): Device for computation (default: DEVICE).

    Returns:
    - tuple: If predict_only is True, returns a tuple with predictions; if False, returns a tuple with predictions, true labels, and probabilities.
    """
    model.eval()

    predictions = []

    if not predict_only:
        reference, probabilities = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            imgs = batch.to(device) if predict_only else batch[0].to(device)

            logits = model(imgs).data
            y_hat = torch.argmax(logits, 1)

            predictions.extend(y_hat.cpu().numpy())

            if not predict_only:
                labels = batch[1].to(device)
                reference.extend(labels.cpu().numpy())
                probabilities.extend(torch.nn.functional.softmax(
                    logits, dim=1)[:, 1].cpu().numpy())

            print(f'\r{i+1}/{len(dataloader)} {print_progress_bar(i, len(dataloader))}', end='', flush=True)

        print()
        if predict_only:
            return predictions,

        return predictions, reference, probabilities


def build_dataloader(dataset: Dataset, batch_size: int = BATCH_SIZE, n_workers: int = N_WORKERS, shuffle: bool = True):
    """
    Builds a DataLoader for a given dataset.

    Args:
    - dataset (Dataset): PyTorch Dataset.
    - batch_size (int): Batch size for the DataLoader (default: BATCH_SIZE).
    - n_workers (int): Number of workers for data loading (default: N_WORKERS).
    - shuffle (bool): Whether to shuffle the data
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
