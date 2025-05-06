import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.manifold import TSNE

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        z = F.relu(self.fc1(x))  # Hidden layer with ReLU activation
        x = self.fc2(z)  # Output layer
        return x, z  # Return both the output and hidden features

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs, _ = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Store misclassified images
            misclassified_mask = (predicted != target)
            if misclassified_mask.any():
                misclassified_images.append(data[misclassified_mask])
                misclassified_labels.append(target[misclassified_mask])
                misclassified_preds.append(predicted[misclassified_mask])
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    # Concatenate misclassified data
    if misclassified_images:
        misclassified_images = torch.cat(misclassified_images, dim=0)
        misclassified_labels = torch.cat(misclassified_labels, dim=0)
        misclassified_preds = torch.cat(misclassified_preds, dim=0)
    
    return test_loss, test_acc, misclassified_images, misclassified_labels, misclassified_preds

def plot_training_curves(train_losses, test_losses, title="Training and Test Losses"):
    """Plot the training and test loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()

def plot_misclassified_images(images, true_labels, pred_labels, n=10):
    """Plot some misclassified images"""
    plt.figure(figsize=(15, 3))
    for i in range(min(n, len(images))):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
        plt.title(f'True: {true_labels[i].item()}\nPred: {pred_labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_images.png')
    plt.close()

def visualize_tsne(features, labels, title, filename):
    """Create a t-SNE visualization of the features"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.detach().cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels.cpu().numpy(), 
                         cmap='tab10', alpha=0.7, s=5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def run_experiment(hidden_size=128, batch_size=64, learning_rate=0.001, num_epochs=10, 
                  seed=42, use_validation=False, validation_size=10000):
    """Run a complete MNIST experiment with the specified parameters"""
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if use_validation:
        # Split the training data to create a validation set
        val_size = validation_size
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        # Create data loaders without validation set
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = None
    
    # Initialize the model
    model = TwoLayerNN(input_size=28*28, hidden_size=hidden_size, output_size=10).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    best_test_loss = None
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        test_loss, test_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        # Evaluate on validation set if used
        if use_validation:
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Load the best model if validation was used
    if use_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Final evaluation
    final_test_loss, final_test_acc, misclassified_images, misclassified_labels, misclassified_preds = evaluate(
        model, test_loader, criterion, device)
    
    if use_validation:
        final_val_loss, final_val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        print(f'Final results - Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.2f}%, '
              f'Test Loss: {final_test_loss:.4f}, Test Acc: {final_test_acc:.2f}%')
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_curves_seed_{seed}.png')
        plt.close()
    else:
        print(f'Final results - Test Loss: {final_test_loss:.4f}, Test Acc: {final_test_acc:.2f}%')
        
        # Plot training curves
        plot_training_curves(train_losses, test_losses, title=f'Training and Test Losses (Seed {seed})')
    
    # Plot misclassified images
    if len(misclassified_images) > 0:
        plot_misclassified_images(misclassified_images, misclassified_labels, misclassified_preds)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'val_losses': val_losses if use_validation else None,
        'final_test_loss': final_test_loss,
        'final_test_acc': final_test_acc,
        'best_val_loss': best_val_loss if use_validation else None,
        'best_test_loss': best_test_loss if use_validation else None,
        'device': device
    }

def feature_analysis(model, dataset, device, filename_prefix="tsne"):
    """Perform t-SNE analysis of the features"""
    # Create a data loader for feature extraction
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    
    # Extract features
    model.eval()
    all_inputs = []
    all_hidden = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            inputs = data.view(-1, 28*28)
            outputs, hidden = model(data)
            
            all_inputs.append(inputs)
            all_hidden.append(hidden)
            all_labels.append(target)
    
    # Concatenate all data
    all_inputs = torch.cat(all_inputs, dim=0)
    all_hidden = torch.cat(all_hidden, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Subsample for t-SNE (it can be slow for large datasets)
    n_samples = min(5000, len(all_labels))
    indices = torch.randperm(len(all_labels))[:n_samples]
    
    sampled_inputs = all_inputs[indices]
    sampled_hidden = all_hidden[indices]
    sampled_labels = all_labels[indices]
    
    # Visualize input features
    visualize_tsne(sampled_inputs, sampled_labels, 
                  "t-SNE Visualization of Input Features", 
                  f"{filename_prefix}_input_features.png")
    
    # Visualize hidden features
    visualize_tsne(sampled_hidden, sampled_labels,
                  "t-SNE Visualization of Hidden Features",
                  f"{filename_prefix}_hidden_features.png")
    
    return sampled_inputs, sampled_hidden, sampled_labels
