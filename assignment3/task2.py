import sys
import os
print(f"Python interpreter: {sys.executable}")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

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
    Load MNIST dataset with controlled randomness
    """
    # Set seed before data loading
    set_seed(seed)
    
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())

    # Data loader with fixed seed
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    return train_loader, test_loader

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
    Train the model with a specific seed
    """
    # Set seed for this run
    set_seed(seed)
    
    # Load data with seed
    train_loader, test_loader = load_data(seed)
    
    # Create model
    model = create_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store errors for plotting
    train_errors = []
    test_errors = []
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
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
        print(f'Seed {seed}, Epoch [{epoch+1}/{num_epochs}], Train Error: {train_error:.4f}, '
              f'Test Error: {test_error:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    # Final evaluation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = 100 * correct / total
    final_error = 100 - final_accuracy
    print(f'Seed {seed}, Final Test Accuracy: {final_accuracy:.2f}%, Final Test Error: {final_error:.2f}%')
    
    return {
        'train_errors': train_errors,
        'test_errors': test_errors,
        'final_accuracy': final_accuracy,
        'final_error': final_error
    }


def plot_multiple_runs(all_results):
    """
    Plot test errors from multiple runs with different seeds
    """
    plt.figure(figsize=(12, 6))
    
    # Plot test errors for each seed
    for i, (seed, results) in enumerate(all_results.items()):
        plt.plot(range(1, num_epochs+1), results['test_errors'], label=f'Seed {seed}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.title('Test Errors for Different Seeds')
    plt.legend()
    plt.grid(True)
    plt.savefig('task2_test_errors_multiple_seeds2.png')
    plt.close()
    
    # Calculate and report statistics
    final_errors = [results['final_error'] for results in all_results.values()]
    mean_error = np.mean(final_errors)
    std_error = np.std(final_errors)
    
    print("\nFinal Test Errors Statistics:")
    print(f"Mean Final Test Error: {mean_error:.2f}%")
    print(f"Standard Deviation of Final Test Error: {std_error:.2f}%")
    print(f"Coefficient of Variation: {(std_error/mean_error)*100:.2f}%")
    
    # Analyze robustness
    if std_error < 0.5:  # This is an arbitrary threshold; adjust based on domain knowledge
        robustness = "The model appears to be robust to seed choice, with low variation in final test errors."
    elif std_error < 1.0:
        robustness = "The model shows moderate sensitivity to seed choice."
    else:
        robustness = "The model shows high sensitivity to seed choice, suggesting potential instability."
    
    print("\nRobustness Analysis:")
    print(robustness)
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'final_errors': final_errors,
        'robustness': robustness
    }


def run_main():
    """
    Main function to run the experiment with multiple seeds
    """
    print("Task 2: Running model with multiple seeds to evaluate robustness")
    all_results = {}
    
    # Train model with each seed
    for seed in seeds:
        print(f"\n--- Starting run with seed {seed} ---")
        results = train_model(seed)
        all_results[seed] = results
    
    # Plot and analyze results
    stats = plot_multiple_runs(all_results)
    
    # Save statistics to file
    with open('task2_statistics.txt', 'w') as f:
        f.write("Final Test Errors Statistics:\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Final errors for each seed: {stats['final_errors']}\n")
        f.write(f"Mean Final Test Error: {stats['mean_error']:.2f}%\n")
        f.write(f"Standard Deviation of Final Test Error: {stats['std_error']:.2f}%\n")
        f.write(f"Coefficient of Variation: {(stats['std_error']/stats['mean_error'])*100:.2f}%\n\n")
        f.write("Robustness Analysis:\n")
        f.write(f"{stats['robustness']}\n")


if __name__ == "__main__":
    run_main()