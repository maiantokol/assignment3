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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Define the neural network model
class TwoLayerNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        z = torch.relu(self.fc1(x))  # Hidden layer with ReLU activation
        x = self.fc2(z)  # Output layer
        return x, z  # Return both output and hidden features


# Task 1: Basic MNIST Classification
def task1(seed=42):
    print("\n=== Task 1: Basic MNIST Classification ===")
    
    # Set seed
    set_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # Initialize the model, loss function, and optimizer
    model = TwoLayerNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    test_losses = []
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
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
        misclassified_images = []
        misclassified_labels = []
        misclassified_preds = []
        
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()
    open_image('loss_curves.png')
    
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
    
    print(f"Final test loss: {test_losses[-1]:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print("Results saved as 'loss_curves.png' and 'misclassified_images.png'")
    
    return model, train_losses, test_losses


# Task 2: Mitigating Pseudorandomness
def task2():
    print("\n=== Task 2: Mitigating Pseudorandomness ===")
    
    seeds = [42, 123, 456, 789, 1024]
    all_test_losses = []
    all_accuracies = []
    
    plt.figure(figsize=(10, 6))
    
    for i, seed in enumerate(seeds):
        print(f"\nRunning experiment with seed {seed}")
        
        # Set seed
        set_seed(seed)
        
        # Run training
        model, _, test_losses = task1(seed)
        
        # Store results
        all_test_losses.append(test_losses[-1])  # Final test loss
        
        # Plot test loss curves for this seed
        plt.plot(range(1, len(test_losses) + 1), test_losses, label=f'Seed {seed}')
    
    # Compute statistics
    mean_loss = np.mean(all_test_losses)
    std_loss = np.std(all_test_losses)
    
    # Finalize and save plot
    plt.title('Test Loss Curves for Different Seeds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('seed_test_loss_curves.png')
    plt.close()
    open_image('seed_test_loss_curves.png')
    
    print("\n=== Seed Test Results ===")
    print(f"Seeds: {seeds}")
    print(f"Final test losses: {[f'{loss:.4f}' for loss in all_test_losses]}")
    print(f"Mean test loss: {mean_loss:.4f}")
    print(f"Standard deviation: {std_loss:.4f}")
    print(f"Coefficient of variation: {(std_loss/mean_loss):.4f}")


# Task 3: Validation Dataset
def task3():
    print("\n=== Task 3: Validation Dataset ===")
    
    seeds = [42, 123, 456, 789, 1024]
    validation_size = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    all_best_val_losses = []
    all_corresponding_test_losses = []
    
    for seed in seeds:
        print(f"\nRunning experiment with seed {seed}")
        set_seed(seed)
        
        # Load datasets
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Split training set into train and validation
        train_size = len(full_train_dataset) - validation_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1000)
        test_loader = DataLoader(test_dataset, batch_size=1000)
        
        # Initialize model
        model = TwoLayerNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training variables
        num_epochs = 10
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
        
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        print(f"Corresponding test loss: {best_test_loss:.4f}")
    
    # Create bar chart
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
    
    print("\n=== Validation Results ===")
    print(f"Best validation losses: {[f'{loss:.4f}' for loss in all_best_val_losses]}")
    print(f"Corresponding test losses: {[f'{loss:.4f}' for loss in all_corresponding_test_losses]}")
    print(f"Mean best validation loss: {np.mean(all_best_val_losses):.4f}")
    print(f"Mean corresponding test loss: {np.mean(all_corresponding_test_losses):.4f}")


# Task 4: Grid Search
def task4():
    print("\n=== Task 4: Grid Search ===")
    
    # Define hyperparameter values to search
    hidden_sizes = [64, 128, 256]
    batch_sizes = [32, 128]
    learning_rates = [0.01, 0.001, 0.0001]
    
    # Set a fixed seed for reproducibility across all experiments
    seed = 42
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training set into train and validation
    validation_size = 10000
    train_size = len(full_train_dataset) - validation_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size])
    
    # Store results
    results = []
    total_combinations = len(hidden_sizes) * len(batch_sizes) * len(learning_rates)
    current_combination = 0
    
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                current_combination += 1
                print(f"\nCombination {current_combination}/{total_combinations}:")
                print(f"hidden_size={hidden_size}, batch_size={batch_size}, learning_rate={lr}")
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                
                # Initialize model
                model = TwoLayerNN(hidden_size=hidden_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Training variables
                num_epochs = 15
                best_val_loss = float('inf')
                best_model_state = None
                best_test_loss = None
                best_test_acc = None
                
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
                    
                    print(f'Epoch {epoch+1}/{num_epochs}: '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
                    
                    # Save the best model based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_test_loss = test_loss
                        best_test_acc = test_acc
                        best_model_state = model.state_dict().copy()
                
                # Store the results
                results.append({
                    'hidden_size': hidden_size,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'val_loss': best_val_loss,
                    'test_loss': best_test_loss,
                    'test_acc': best_test_acc
                })
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'])
    
    # Display results
    print("\n===== Grid Search Results =====")
    print(f"{'Hidden Size':<15} {'Batch Size':<15} {'Learning Rate':<15} {'Val Loss':<15} {'Test Loss':<15} {'Test Acc':<15}")
    
    for result in results:
        print(f"{result['hidden_size']:<15} {result['batch_size']:<15} {result['learning_rate']:<15.4f} "
              f"{result['val_loss']:<15.4f} {result['test_loss']:<15.4f} {result['test_acc']:<15.2f}")
    
    # Print the best combination
    best = results[0]
    print("\n===== Best Combination =====")
    print(f"Hidden Size: {best['hidden_size']}")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Learning Rate: {best['learning_rate']:.4f}")
    print(f"Validation Loss: {best['val_loss']:.4f}")
    print(f"Test Loss: {best['test_loss']:.4f}")
    print(f"Test Accuracy: {best['test_acc']:.2f}%")
    
    # Create a heatmap visualization (simplified version)
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 8))
    
    # Create a simple visualization
    for i, hs in enumerate(hidden_sizes):
        for j, bs in enumerate(batch_sizes):
            val_losses = [r['val_loss'] for r in results if r['hidden_size'] == hs and r['batch_size'] == bs]
            lrs = [r['learning_rate'] for r in results if r['hidden_size'] == hs and r['batch_size'] == bs]
            plt.plot(range(len(val_losses)), val_losses, 
                    marker='o', label=f'HS={hs}, BS={bs}')
    
    plt.title('Validation Losses for Different Hyperparameter Combinations')
    plt.xlabel('Learning Rate Index (0=0.01, 1=0.001, 2=0.0001)')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('grid_search_results.png')
    plt.close()
    open_image('grid_search_results.png')


# Task 5: Feature Analysis
def task5():
    print("\n=== Task 5: Feature Analysis ===")
    
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the best parameters from Task 4 (assumed values)
    hidden_size = 128  # Replace with the best value from Task 4
    batch_size = 64
    learning_rate = 0.001
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train a model with the best parameters
    print("Training a model for feature analysis...")
    model = TwoLayerNN(hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
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
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
    
    # Extract features
    print("\nExtracting features for t-SNE visualization...")
    from sklearn.manifold import TSNE
    
    # Create a subset for t-SNE analysis (t-SNE is computationally intensive)
    subset_size = 2000
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=200, shuffle=False)
    
    # Extract input features and hidden features
    all_inputs = []
    all_hidden = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            inputs = images.view(-1, 28*28)
            outputs, hidden = model(images)
            
            all_inputs.append(inputs.cpu())
            all_hidden.append(hidden.cpu())
            all_labels.append(labels.cpu())
    
    all_inputs = torch.cat(all_inputs).numpy()
    all_hidden = torch.cat(all_hidden).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Apply t-SNE to input features
    print("Applying t-SNE to input features...")
    tsne_input = TSNE(n_components=2, random_state=seed)
    input_2d = tsne_input.fit_transform(all_inputs)
    
    # Apply t-SNE to hidden features
    print("Applying t-SNE to hidden features...")
    tsne_hidden = TSNE(n_components=2, random_state=seed)
    hidden_2d = tsne_hidden.fit_transform(all_hidden)
    
    # Plot t-SNE visualizations
    plt.figure(figsize=(20, 10))
    
    # Plot input features
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(input_2d[:, 0], input_2d[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Input Features')
    
    # Plot hidden features
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Hidden Features')
    
    plt.tight_layout()
    plt.savefig('tsne_visualizations.png')
    plt.close()
    open_image('tsne_visualizations.png')
    
    print("t-SNE visualizations saved as 'tsne_visualizations.png'")


# Main function to run all tasks
def run_all_tasks():
    task1()
    task2()
    task3()
    task4()
    task5()


if __name__ == "__main__":
    print("MNIST Classification with PyTorch")
    print("Choose a task to run (1-5) or 'all' to run all tasks:")
    print("1: Basic MNIST Classification")
    print("2: Mitigating Pseudorandomness")
    print("3: Validation Dataset")
    print("4: Grid Search")
    print("5: Feature Analysis")
    
    choice = input("Enter your choice: ")
    
    if choice == "1":
        task1()
    elif choice == "2":
        task2()
    elif choice == "3":
        task3()
    elif choice == "4":
        task4()
    elif choice == "5":
        task5()
    elif choice.lower() == "all":
        run_all_tasks()
    else:
        print("Invalid choice. Exiting.")
