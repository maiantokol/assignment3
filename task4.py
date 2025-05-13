import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
import itertools
from datetime import datetime

# Fixed parameters
input_size = 784
num_classes = 10
num_epochs = 5
val_size = 10000  # Validation set size

# Grid search hyperparameters
hidden_sizes = [100, 300, 500]  # Size of the hidden layer
batch_sizes = [50, 100, 200]    # Batch size for training
learning_rates = [0.01, 0.001, 0.0001]  # Learning rate

# Use a single seed for grid search to save computation time
seed = 42

def set_seed(seed_value):
    """
    Set seed for reproducibility
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def load_data(seed, batch_size):
    """
    Load MNIST dataset and split into train, validation, and test sets
    """
    # Set seed before data loading
    set_seed(seed)
    
    # MNIST dataset
    full_train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())
    
    # Split training set to create validation set
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)  # Use the same seed for consistent splits
    )
    
    # Data loaders with specified batch size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    return train_loader, val_loader, test_loader

# Initialize model
def create_model(hidden_size):
    model = NeuralNet(input_size, hidden_size, num_classes)
    return model

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def train_model_with_params(hidden_size, batch_size, learning_rate):
    """
    Train a model with specific hyperparameters
    """
    print(f"\nTraining with hyperparameters: hidden_size={hidden_size}, batch_size={batch_size}, learning_rate={learning_rate}")
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load data with specified batch size
    train_loader, val_loader, test_loader = load_data(seed, batch_size)
    
    # Create model with specified hidden size
    model = create_model(hidden_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store errors for plotting
    train_errors = []
    val_errors = []
    test_errors = []
    
    # Keep track of best model and minimum validation error
    min_val_error = float('inf')
    best_test_error = None
    best_epoch = -1
    best_model_state = None
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Print progress less frequently to reduce output
            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
        
        # Calculate training error for the epoch
        train_error = running_loss / len(train_loader)
        train_errors.append(train_error)

        # Calculate validation error for the epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, input_size)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        val_error = val_loss / len(val_loader)
        val_errors.append(val_error)
        
        # Calculate test error for the epoch
        test_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, input_size)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_error = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        test_errors.append(test_error)
        
        # Check if this is the best model based on validation error
        if val_error < min_val_error:
            min_val_error = val_error
            best_test_error = test_error
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Error: {train_error:.4f}, '
              f'Val Error: {val_error:.4f}, Test Error: {test_error:.4f}')
    
    # Load the best model based on validation error
    model.load_state_dict(best_model_state)
    
    # Final evaluation of the best model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    best_test_accuracy = 100 * correct / total
    best_test_error_pct = 100 - best_test_accuracy
    
    print(f"\nBest Model (Epoch {best_epoch+1}):")
    print(f"Minimum Validation Error: {min_val_error:.4f}")
    print(f"Corresponding Test Error: {best_test_error:.4f}")
    print(f"Final Test Accuracy: {best_test_accuracy:.2f}%, Test Error: {best_test_error_pct:.2f}%")
    
    return {
        'hidden_size': hidden_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_errors': train_errors,
        'val_errors': val_errors,
        'test_errors': test_errors,
        'min_val_error': min_val_error,
        'best_test_error': best_test_error,
        'best_epoch': best_epoch,
        'best_test_accuracy': best_test_accuracy,
        'best_test_error_pct': best_test_error_pct
    }


def plot_multiple_runs(all_results):
    """
    Plot validation and corresponding test errors from multiple runs with different seeds
    """
    # Create figure for error curves
    plt.figure(figsize=(12, 6))
    
    # Plot validation errors for each seed
    for i, (seed, results) in enumerate(all_results.items()):
        plt.plot(range(1, num_epochs+1), results['val_errors'], linestyle='-', marker='o',
                 label=f'Seed {seed} - Validation Error')
    
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Validation Errors for Different Seeds')
    plt.legend()
    plt.grid(True)
    plt.savefig('task3_validation_errors.png')
    plt.close()
    
    # Create scatter plot of min validation error vs corresponding test error
    plt.figure(figsize=(10, 6))
    min_val_errors = [results['min_val_error'] for results in all_results.values()]
    best_test_errors = [results['best_test_error'] for results in all_results.values()]
    
    # Plot points
    plt.scatter(min_val_errors, best_test_errors, s=100, c='blue', alpha=0.7)
    
    # Label points with seed values
    for i, (seed, results) in enumerate(all_results.items()):
        plt.annotate(f'Seed {seed}', 
                     (results['min_val_error'], results['best_test_error']),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Minimum Validation Error')
    plt.ylabel('Corresponding Test Error')
    plt.title('Minimum Validation Error vs Corresponding Test Error')
    plt.grid(True)
    plt.savefig('task3_min_val_vs_test_error.png')
    plt.close()
    
    # Calculate and report statistics
    min_val_errors = [results['min_val_error'] for results in all_results.values()]
    corresponding_test_errors = [results['best_test_error'] for results in all_results.values()]
    best_epochs = [results['best_epoch'] + 1 for results in all_results.values()]  # +1 for 1-indexing
    
    mean_val_error = np.mean(min_val_errors)
    std_val_error = np.std(min_val_errors)
    
    mean_test_error = np.mean(corresponding_test_errors)
    std_test_error = np.std(corresponding_test_errors)
    
    print("\nStatistics for models selected by minimum validation error:")
    print(f"Mean Minimum Validation Error: {mean_val_error:.4f} ± {std_val_error:.4f}")
    print(f"Mean Corresponding Test Error: {mean_test_error:.4f} ± {std_test_error:.4f}")
    print(f"Best Epochs: {best_epochs}")
    print(f"Mean Best Epoch: {np.mean(best_epochs):.1f}")
    
    # Create table of results
    results_table = []
    results_table.append("| Seed | Best Epoch | Min Val Error | Corresponding Test Error |")
    results_table.append("| ---- | ---------- | ------------- | ------------------------ |")
    
    for seed, results in all_results.items():
        results_table.append(f"| {seed} | {results['best_epoch']+1} | {results['min_val_error']:.4f} | {results['best_test_error']:.4f} |")
    
    results_table.append(f"| **Mean** | **{np.mean(best_epochs):.1f}** | **{mean_val_error:.4f}** | **{mean_test_error:.4f}** |")
    results_table.append(f"| **Std** | **{np.std(best_epochs):.1f}** | **{std_val_error:.4f}** | **{std_test_error:.4f}** |")
    
    # Save results table to file
    with open('task3_results_table.md', 'w') as f:
        f.write('\n'.join(results_table))
    
    return {
        'mean_val_error': mean_val_error,
        'std_val_error': std_val_error,
        'mean_test_error': mean_test_error,
        'std_test_error': std_test_error,
        'best_epochs': best_epochs,
        'min_val_errors': min_val_errors,
        'corresponding_test_errors': corresponding_test_errors
    }


def run_main():
    """
    Main function to run the experiment with multiple seeds
    """
    print("Task 3: Running models with validation-based early stopping")
    all_results = {}
    
    # Train model with each seed
    for seed in seeds:
        print(f"\n--- Starting run with seed {seed} ---")
        results = train_model(seed)
        all_results[seed] = results
    
    # Plot and analyze results
    stats = plot_multiple_runs(all_results)
    
    # Save statistics to file
    with open('task3_statistics.txt', 'w') as f:
        f.write("Task 3: Model Selection using Validation Set\n")
        f.write("=============================================\n\n")
        f.write(f"Seeds used: {seeds}\n\n")
        f.write("Statistics for models selected by minimum validation error:\n")
        f.write(f"Mean Minimum Validation Error: {stats['mean_val_error']:.4f} ± {stats['std_val_error']:.4f}\n")
        f.write(f"Mean Corresponding Test Error: {stats['mean_test_error']:.4f} ± {stats['std_test_error']:.4f}\n")
        f.write(f"Best Epochs: {stats['best_epochs']}\n")
        f.write(f"Mean Best Epoch: {np.mean(stats['best_epochs']):.1f}\n\n")
        
        f.write("Individual Results:\n")
        for i, seed in enumerate(seeds):
            f.write(f"Seed {seed}:\n")
            f.write(f"  Best Epoch: {stats['best_epochs'][i]}\n")
            f.write(f"  Min Validation Error: {stats['min_val_errors'][i]:.4f}\n")
            f.write(f"  Corresponding Test Error: {stats['corresponding_test_errors'][i]:.4f}\n\n")


def perform_grid_search():
    """
    Perform grid search over hyperparameters
    """
    print("Task 4: Grid Search for Hyperparameter Optimization")
    print(f"Grid Search Parameters:")
    print(f"  - Hidden Sizes: {hidden_sizes}")
    print(f"  - Batch Sizes: {batch_sizes}")
    print(f"  - Learning Rates: {learning_rates}")
    print(f"Using seed: {seed}")
    
    # Time tracking
    start_time = datetime.now()
    print(f"Starting grid search at: {start_time}")
    
    # Create list to store results
    results = []
    
    # Iterate over all combinations
    total_combinations = len(hidden_sizes) * len(batch_sizes) * len(learning_rates)
    current_combination = 1
    
    for hidden_size, batch_size, learning_rate in itertools.product(hidden_sizes, batch_sizes, learning_rates):
        print(f"\nCombination {current_combination}/{total_combinations}:")
        result = train_model_with_params(hidden_size, batch_size, learning_rate)
        results.append(result)
        current_combination += 1
        
        # Save intermediate results in case the process is interrupted
        results_df = pd.DataFrame(results)
        results_df.to_csv('task4_grid_search_results_partial.csv', index=False)
    
    # Create DataFrame from results and sort by validation error
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('min_val_error')
    
    # Find the best hyperparameters
    best_params = results_df.iloc[0]
    
    # Create a nicely formatted table for reporting
    table_df = results_df[['hidden_size', 'batch_size', 'learning_rate', 'min_val_error', 'best_test_error', 'best_epoch']]
    
    # Format for markdown table
    md_table = ["| Hidden Size | Batch Size | Learning Rate | Min Val Error | Test Error | Best Epoch |"]
    md_table.append("| ---------- | ---------- | ------------ | ------------- | ---------- | ---------- |")
    
    # Add rows to table, highlighting the best combination
    for i, row in table_df.iterrows():
        is_best = (row['hidden_size'] == best_params['hidden_size'] and 
                   row['batch_size'] == best_params['batch_size'] and 
                   row['learning_rate'] == best_params['learning_rate'])
        
        row_str = f"| {row['hidden_size']} | {row['batch_size']} | {row['learning_rate']:.6f} | {row['min_val_error']:.4f} | {row['best_test_error']:.4f} | {int(row['best_epoch'])} |"
        
        if is_best:
            # Add bold formatting for best row
            row_str = "**" + row_str + "**"
        
        md_table.append(row_str)
    
    # Save the results
    results_df.to_csv('task4_grid_search_results.csv', index=False)
    
    with open('task4_results_table.md', 'w') as f:
        f.write("## Grid Search Results\n\n")
        f.write("The following table shows the test error corresponding to the minimum validation error for each hyperparameter combination.\n\n")
        f.write("**Note:** The best combination is highlighted in bold.\n\n")
        f.write('\n'.join(md_table))
        f.write("\n\n### Best Hyperparameters\n\n")
        f.write(f"- Hidden Size: {best_params['hidden_size']}\n")
        f.write(f"- Batch Size: {best_params['batch_size']}\n")
        f.write(f"- Learning Rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"- Minimum Validation Error: {best_params['min_val_error']:.4f}\n")
        f.write(f"- Corresponding Test Error: {best_params['best_test_error']:.4f}\n")
        f.write(f"- Best Epoch: {int(best_params['best_epoch'])}\n")
        f.write(f"- Best Test Accuracy: {best_params['best_test_accuracy']:.2f}%\n")
    
    # Visual representations of results
    # Create heatmap for test errors with different hidden_size and learning_rate (for best batch_size)
    best_batch = best_params['batch_size']
    heatmap_data = results_df[results_df['batch_size'] == best_batch].pivot(
        index='hidden_size', 
        columns='learning_rate', 
        values='best_test_error'
    )
    
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Test Error')
    plt.xticks(range(len(heatmap_data.columns)), [f"{lr:.6f}" for lr in heatmap_data.columns])
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.xlabel('Learning Rate')
    plt.ylabel('Hidden Size')
    plt.title(f'Test Error Heatmap (Batch Size = {best_batch})')
    
    # Add text annotations in the cells
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            plt.text(j, i, f"{heatmap_data.iloc[i, j]:.4f}", 
                     ha="center", va="center", color="white")
    
    plt.tight_layout()
    plt.savefig('task4_test_error_heatmap.png')
    plt.close()
    
    # Bar plot comparing different hyperparameter combinations
    plt.figure(figsize=(12, 6))
    
    # Create labels for hyperparameter combinations
    combo_labels = [f"h{row['hidden_size']}_b{row['batch_size']}_lr{row['learning_rate']:.4f}" 
                     for _, row in results_df[:10].iterrows()]
    
    # Get corresponding test errors for top 10 models
    test_errors = results_df['best_test_error'][:10].values
    val_errors = results_df['min_val_error'][:10].values
    
    x = np.arange(len(combo_labels))
    width = 0.35
    
    plt.bar(x - width/2, val_errors, width, label='Validation Error', color='skyblue')
    plt.bar(x + width/2, test_errors, width, label='Test Error', color='orange')
    
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Error')
    plt.title('Top 10 Hyperparameter Combinations by Validation Error')
    plt.xticks(x, combo_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('task4_top_hyperparameters.png')
    plt.close()
    
    # Calculate runtime
    end_time = datetime.now()
    runtime = end_time - start_time
    print(f"\nGrid search completed in: {runtime}")
    print(f"Total combinations evaluated: {total_combinations}")
    print("\nBest Hyperparameters:")
    print(f"Hidden Size: {best_params['hidden_size']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"Minimum Validation Error: {best_params['min_val_error']:.4f}")
    print(f"Corresponding Test Error: {best_params['best_test_error']:.4f}")
    
    return results_df


# Run the grid search
if __name__ == "__main__":
    perform_grid_search()