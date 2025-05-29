import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import subprocess
import platform

# Helper function to open images
def open_image(image_path):
    """Open an image file using the default viewer."""
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', image_path])
        elif platform.system() == 'Windows':
            os.startfile(image_path)
        elif platform.system() == 'Linux':
            subprocess.run(['xdg-open', image_path])
    except Exception as e:
        print(f"Could not open image: {e}")

# Set seed for reproducibility
def set_seed(seed):
    """Set random seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Define the neural network model
class TwoLayerNN(nn.Module):
    """A two-layer neural network for MNIST classification."""
    def __init__(self, hidden_size=128):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        z = torch.relu(self.fc1(x))  # Hidden layer with ReLU activation
        x = self.fc2(z)  # Output layer
        return x, z  # Return both output and hidden features

def get_data_loaders(batch_size=64, validation_size=None):
    """
    Create and return data loaders for MNIST dataset.
    
    Args:
        batch_size (int): Batch size for training
        validation_size (int, optional): If provided, split training set to include validation
        
    Returns:
        Dictionary containing data loaders and dataset sizes
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if validation_size is not None:
        # Split training set into train and validation
        train_size = len(full_train_dataset) - validation_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_size': train_size,
            'val_size': validation_size,
            'test_size': len(test_dataset)
        }
    else:
        # Create data loaders without validation split
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_size': len(full_train_dataset),
            'test_size': len(test_dataset)
        }

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: The neural network model
        data_loader: DataLoader containing evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store incorrect predictions
            incorrect_mask = (predicted != labels)
            if incorrect_mask.any():
                all_outputs.append(predicted[incorrect_mask])
                all_labels.append(labels[incorrect_mask])
    
    # Calculate statistics
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'incorrect_outputs': all_outputs,
        'incorrect_labels': all_labels
    }

def plot_loss_curves(train_losses, test_losses, title='Training and Test Losses', filename='loss_curves.png'):
    """
    Plot training and test loss curves.
    
    Args:
        train_losses: List of training losses
        test_losses: List of test losses
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    open_image(filename)
