"""
Task 3: Validation Dataset

This script uses a validation dataset for early stopping and model selection.
It trains models with different random seeds, using validation loss to select
the best model, then evaluates on the test set.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mnist_classifier import TwoLayerNN, set_seed, get_data_loaders, open_image

def run_task():
    """
    Run Task 3: Using a validation dataset for model selection.
    
    Returns:
        Dictionary of results from the validation experiments
    """
    print("\n=== Task 3: Validation Dataset ===")
    
    # Define constants
    seeds = [42, 123, 456, 789, 1024]
    validation_size = 10000
    batch_size = 64
    num_epochs = 10
    hidden_size = 128
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Store results
    all_best_val_losses = []
    all_corresponding_test_losses = []
    best_epochs = []
    
    # Run experiments with different seeds
    for seed in seeds:
        print(f"\nRunning experiment with seed {seed}")
        set_seed(seed)
        
        # Get data loaders with validation split
        data = get_data_loaders(batch_size=batch_size, validation_size=validation_size)
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        test_loader = data['test_loader']
        
        # Initialize model
        model = TwoLayerNN(hidden_size=hidden_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training variables
        train_losses = []
        val_losses = []
        test_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_test_loss = None
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Test
            test_loss = 0.0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
        
        # Store results
        all_best_val_losses.append(best_val_loss)
        all_corresponding_test_losses.append(best_test_loss)
        best_epochs.append(best_epoch)
        
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        print(f"Corresponding test loss: {best_test_loss:.4f}")
    
    # Create bar chart for visualization
    plt.figure(figsize=(12, 6))
    x = np.arange(len(seeds))
    width = 0.35
    
    plt.bar(x - width/2, all_best_val_losses, width, label='Best Validation Loss')
    plt.bar(x + width/2, all_corresponding_test_losses, width, label='Corresponding Test Loss')
    
    plt.xlabel('Seed')
    plt.ylabel('Loss')
    plt.title('Best Validation Loss and Corresponding Test Loss for Different Seeds')
    plt.xticks(x, [str(s) for s in seeds])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('validation_test_loss_comparison.png')
    plt.close()
    open_image('validation_test_loss_comparison.png')
    
    # Calculate statistics
    mean_val_loss = np.mean(all_best_val_losses)
    mean_test_loss = np.mean(all_corresponding_test_losses)
    
    # Print summary
    print("\n=== Validation Results ===")
    print(f"Best validation losses: {[f'{loss:.4f}' for loss in all_best_val_losses]}")
    print(f"Corresponding test losses: {[f'{loss:.4f}' for loss in all_corresponding_test_losses]}")
    print(f"Mean best validation loss: {mean_val_loss:.4f}")
    print(f"Mean corresponding test loss: {mean_test_loss:.4f}")
    
    return {
        'seeds': seeds,
        'best_val_losses': all_best_val_losses,
        'corresponding_test_losses': all_corresponding_test_losses,
        'best_epochs': best_epochs,
        'mean_val_loss': mean_val_loss,
        'mean_test_loss': mean_test_loss
    }

if __name__ == "__main__":
    run_task()
