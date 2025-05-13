import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.manifold import TSNE
import time

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Use a single seed for feature analysis
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
    
    return train_loader, test_loader, train_dataset, test_dataset

# Neural Network model with feature extraction capability
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes, bias=False)
        
        # For storing intermediate values
        self.input_features = None
        self.hidden_features = None

    def forward(self, x):
        # Store input features (x_i)
        self.input_features = x.detach().clone()
        
        # First layer computation (W^(1)T x_i + b^(1))
        pre_activation = self.fc1(x)
        
        # Apply ReLU activation to get hidden features (z_i)
        hidden = self.relu(pre_activation)
        self.hidden_features = hidden.detach().clone()
        
        # Second layer computation
        output = self.fc2(hidden)
        
        return output
    
    def get_hidden_features(self):
        return self.hidden_features
    
    def get_input_features(self):
        return self.input_features

# Initialize model
def create_model():
    model = NeuralNet(input_size, hidden_size, num_classes)
    return model
    
def train_model():
    """
    Train the model with a fixed seed
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load data
    train_loader, test_loader, _, _ = load_data(seed)
    
    # Create model
    model = create_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training model with seed {seed}...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Test the model
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

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
    
    return model


def extract_features(model, dataset, num_samples=5000):
    """
    Extract input features (x_i) and hidden features (z_i) from the model
    using samples from the dataset
    """
    # Create a dataloader with a larger batch size for faster extraction
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    
    model.eval()
    all_hidden_features = []
    all_input_features = []
    all_labels = []
    samples_collected = 0
    
    print(f"Extracting features from {num_samples} samples...")
    with torch.no_grad():
        for images, labels in loader:
            batch_size = images.size(0)
            images = images.reshape(batch_size, -1)  # Flatten images
            
            # Forward pass to extract features
            _ = model(images)
            
            # Get features from the model
            hidden_features = model.get_hidden_features()
            input_features = model.get_input_features()
            
            # Append to the lists
            all_hidden_features.append(hidden_features)
            all_input_features.append(input_features)
            all_labels.append(labels)
            
            samples_collected += batch_size
            if samples_collected >= num_samples:
                break
    
    # Concatenate all collected features and labels
    hidden_features = torch.cat(all_hidden_features, dim=0)[:num_samples]
    input_features = torch.cat(all_input_features, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    
    print(f"Extracted {hidden_features.size(0)} samples")
    print(f"Hidden features shape: {hidden_features.size()}")
    print(f"Input features shape: {input_features.size()}")
    
    return hidden_features, input_features, labels


def apply_tsne(features, perplexity=30, n_iter=1000):
    """
    Apply t-SNE dimensionality reduction to the features
    """
    print(f"Applying t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    start_time = time.time()
    
    # Convert to numpy for sklearn
    features_np = features.cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=seed)
    features_tsne = tsne.fit_transform(features_np)
    
    elapsed_time = time.time() - start_time
    print(f"t-SNE completed in {elapsed_time:.2f} seconds")
    
    return features_tsne


def plot_tsne(features_tsne, labels, title, filename):
    """
    Create a scatter plot of t-SNE embeddings colored by label
    """
    labels_np = labels.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                         c=labels_np, cmap='tab10', alpha=0.7, s=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Digit')
    
    # Set title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add grid
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filename}")
    



def run_feature_analysis():
    """
    Main function to perform feature analysis with t-SNE
    """
    print("Task 5: Feature Analysis with t-SNE Visualization")
    
    # Step 1: Train the model
    model = train_model()
    
    # Step 2: Load full train dataset for feature extraction
    _, _, train_dataset, _ = load_data(seed)
    
    # Step 3: Extract features from train dataset
    hidden_features, input_features, labels = extract_features(model, train_dataset)
    
    # Step 4: Apply t-SNE to hidden features (z_i)
    print("\nProcessing hidden features (z_i = σ(W^(1)T x_i + b^(1)))...")
    hidden_tsne = apply_tsne(hidden_features)
    
    # Step 5: Apply t-SNE to input features (x_i)
    print("\nProcessing input features (x_i)...")
    input_tsne = apply_tsne(input_features)
    
    # Step 6: Create visualizations
    print("\nCreating visualizations...")
    plot_tsne(hidden_tsne, labels, 
             "t-SNE Visualization of Hidden Features (z_i)", 
             "task5_hidden_features_tsne.png")
    
    plot_tsne(input_tsne, labels, 
             "t-SNE Visualization of Input Features (x_i)", 
             "task5_input_features_tsne.png")
    
    print("\nFeature analysis complete! The visualizations show how the neural network")
    print("transforms the input space (x_i) to the hidden representation (z_i).")
    print("The hidden features visualization should show better class separation")
    print("if the model has learned useful representations of the digits.")


if __name__ == "__main__":
    run_feature_analysis()