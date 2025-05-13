import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
val_size = 10000  # Validation set size

# Seeds for different runs
seeds = [42, 123, 456, 789, 999]

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


def load_data(seed):
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
    
    # Data loaders
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
def create_model():
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
    
def train_model(seed):
    """
    Train the model with a specific seed, track validation error,
    and return the test error corresponding to the minimum validation error
    """
    # Set seed for this run
    set_seed(seed)
    
    # Load data with seed
    train_loader, val_loader, test_loader = load_data(seed)
    
    # Create model
    model = create_model()
    
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
            if (i+1) % 100 == 0:
                print(f'Seed {seed}, Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
        
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
            
        print(f'Seed {seed}, Epoch [{epoch+1}/{num_epochs}], Train Error: {train_error:.4f}, '
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
    
    print(f'\nSeed {seed}, Best Model (Epoch {best_epoch+1}):')
    print(f'Minimum Validation Error: {min_val_error:.4f}')
    print(f'Corresponding Test Error: {best_test_error:.4f}')
    print(f'Final Test Accuracy: {best_test_accuracy:.2f}%, Test Error: {best_test_error_pct:.2f}%')
    
    return {
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


if __name__ == "__main__":
    run_main()