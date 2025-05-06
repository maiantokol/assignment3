"""
Task 4: Grid Search

This script performs a grid search over hyperparameters (hidden size,
batch size, learning rate) to find the best model configuration.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mnist_classifier import TwoLayerNN, set_seed, get_data_loaders, open_image

def run_task():
    """
    Run Task 4: Grid search for hyperparameter optimization.
    
    Returns:
        Dictionary containing the grid search results
    """
    print("\n=== Task 4: Grid Search ===")
    
    # Define hyperparameter values to search
    hidden_sizes = [64, 128, 256]
    batch_sizes = [32, 128]
    learning_rates = [0.01, 0.001, 0.0001]
    
    # Set a fixed seed for reproducibility across all experiments
    seed = 42
    set_seed(seed)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation_size = 10000
    num_epochs = 15
    
    # Get data with validation split
    data = get_data_loaders(batch_size=32, validation_size=validation_size)  # Start with a default batch size
    
    # Store results
    results = []
    total_combinations = len(hidden_sizes) * len(batch_sizes) * len(learning_rates)
    current_combination = 0
    
    # Grid search
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            # Create data loaders with current batch size
            data = get_data_loaders(batch_size=batch_size, validation_size=validation_size)
            train_loader = data['train_loader']
            val_loader = data['val_loader']
            test_loader = data['test_loader']
            
            for lr in learning_rates:
                current_combination += 1
                print(f"\nCombination {current_combination}/{total_combinations}:")
                print(f"hidden_size={hidden_size}, batch_size={batch_size}, learning_rate={lr}")
                
                # Initialize model
                model = TwoLayerNN(hidden_size=hidden_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                # Training variables
                best_val_loss = float('inf')
                best_model_state = None
                best_test_loss = None
                best_test_acc = None
                best_epoch = 0
                
                # Training loop
                for epoch in range(num_epochs):
                    # Training
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs, _ = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    train_loss = running_loss / len(train_loader)
                    train_acc = 100 * correct / total
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs, _ = model(images)
                            loss = criterion(outputs, labels)
                            
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    
                    val_loss /= len(val_loader)
                    val_acc = 100 * correct / total
                    
                    # Test
                    test_loss = 0.0
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
                    
                    test_loss /= len(test_loader)
                    test_acc = 100 * correct / total
                    
                    # Print progress
                    print(f'Epoch {epoch+1}/{num_epochs}: '
                          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
                    
                    # Save the best model based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_test_loss = test_loss
                        best_test_acc = test_acc
                        best_model_state = model.state_dict().copy()
                        best_epoch = epoch
                
                # Store the results for this combination
                results.append({
                    'hidden_size': hidden_size,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'val_loss': best_val_loss,
                    'test_loss': best_test_loss,
                    'test_acc': best_test_acc,
                    'best_epoch': best_epoch
                })
                
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                print(f"Corresponding test accuracy: {best_test_acc:.2f}%")
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Sort results by validation loss
    results_df = results_df.sort_values('val_loss')
    
    # Save results to CSV
    results_df.to_csv('grid_search_results.csv', index=False)
    
    # Display results
    print("\n===== Grid Search Results =====")
    print(f"{'Hidden Size':<15} {'Batch Size':<15} {'Learning Rate':<15} {'Val Loss':<15} {'Test Loss':<15} {'Test Acc':<15}")
    
    for _, row in results_df.iterrows():
        print(f"{row['hidden_size']:<15} {row['batch_size']:<15} {row['learning_rate']:<15.4f} "
              f"{row['val_loss']:<15.4f} {row['test_loss']:<15.4f} {row['test_acc']:<15.2f}")
    
    # Print the best combination
    best = results_df.iloc[0]
    print("\n===== Best Combination =====")
    print(f"Hidden Size: {best['hidden_size']}")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Learning Rate: {best['learning_rate']:.4f}")
    print(f"Validation Loss: {best['val_loss']:.4f}")
    print(f"Test Loss: {best['test_loss']:.4f}")
    print(f"Test Accuracy: {best['test_acc']:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot validation loss for different hyperparameter combinations
    for hs in hidden_sizes:
        for bs in batch_sizes:
            mask = (results_df['hidden_size'] == hs) & (results_df['batch_size'] == bs)
            if mask.any():
                subset = results_df[mask].sort_values('learning_rate')
                plt.plot(range(len(subset)), subset['val_loss'], 
                        marker='o', label=f'HS={hs}, BS={bs}')
    
    plt.title('Validation Losses for Different Hyperparameter Combinations')
    plt.xlabel('Learning Rate Index (0=0.01, 1=0.001, 2=0.0001)')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('grid_search_results.png')
    plt.close()
    open_image('grid_search_results.png')
    
    # Create heatmap for better visualization
    try:
        import seaborn as sns
        
        # Reshape data for heatmap
        pivot_data = []
        for hs in hidden_sizes:
            for bs in batch_sizes:
                for lr in learning_rates:
                    mask = (results_df['hidden_size'] == hs) & \
                           (results_df['batch_size'] == bs) & \
                           (results_df['learning_rate'] == lr)
                    if mask.any():
                        val_loss = results_df.loc[mask, 'val_loss'].values[0]
                        pivot_data.append({
                            'Hidden Size': hs,
                            'Batch Size': bs,
                            'Learning Rate': lr,
                            'Validation Loss': val_loss
                        })
        
        pivot_df = pd.DataFrame(pivot_data)
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        for i, bs in enumerate(batch_sizes):
            plt.subplot(len(batch_sizes), 1, i+1)
            subset = pivot_df[pivot_df['Batch Size'] == bs]
            pivot = subset.pivot('Hidden Size', 'Learning Rate', 'Validation Loss')
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
            plt.title(f'Validation Loss Heatmap (Batch Size = {bs})')
        
        plt.tight_layout()
        plt.savefig('grid_search_heatmap.png')
        plt.close()
        open_image('grid_search_heatmap.png')
    except (ImportError, Exception) as e:
        print(f"Could not create heatmap visualization: {e}")
    
    return {
        'results': results,
        'best': best,
        'best_hidden_size': best['hidden_size'],
        'best_batch_size': best['batch_size'],
        'best_learning_rate': best['learning_rate']
    }

if __name__ == "__main__":
    run_task()
