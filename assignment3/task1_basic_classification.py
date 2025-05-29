"""
Task 1: Basic MNIST Classification

This script implements a basic two-layer neural network for MNIST digit classification.
It trains the model, evaluates it, and visualizes results.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mnist_classifier import TwoLayerNN, set_seed, get_data_loaders, evaluate_model, open_image, plot_loss_curves

def run_task(seed=42):
    """
    Run Task 1: Basic MNIST Classification.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Trained model and training history
    """
    print("\n=== Task 1: Basic MNIST Classification ===")
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    data = get_data_loaders(batch_size=64)
    train_loader = data['train_loader']
    test_loader = data['test_loader']
    
    # Initialize the model, loss function, and optimizer
    model = TwoLayerNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training variables
    num_epochs = 10
    train_losses = []
    test_losses = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Get misclassified examples
                mask = (predicted != labels)
                if mask.any():
                    misclassified_images.append(images[mask])
                    misclassified_labels.append(labels[mask])
                    misclassified_preds.append(predicted[mask])
        
        test_loss /= len(test_loader)
        test_accuracy = 100 * correct / total
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    
    # Plot training and test loss curves
    plot_loss_curves(train_losses, test_losses)
    
    # Plot misclassified images
    if misclassified_images:
        misclassified_images = torch.cat(misclassified_images)[:10]  # Take first 10
        misclassified_labels = torch.cat(misclassified_labels)[:10]
        misclassified_preds = torch.cat(misclassified_preds)[:10]
        
        plt.figure(figsize=(15, 3))
        for i in range(10):  # Display 10 misclassified images
            plt.subplot(1, 10, i + 1)
            plt.imshow(misclassified_images[i].cpu().squeeze().numpy(), cmap='gray')
            plt.title(f'True: {misclassified_labels[i].item()}\nPred: {misclassified_preds[i].item()}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('misclassified_images.png')
        plt.close()
        open_image('misclassified_images.png')
    
    print(f"\nFinal test loss: {test_losses[-1]:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print("Results saved as 'loss_curves.png' and 'misclassified_images.png'")
    
    return model, train_losses, test_losses

if __name__ == "__main__":
    run_task()
