"""
Task 5: Feature Analysis

This script analyzes the features learned by the neural network using t-SNE
visualization to compare input features and learned hidden representations.
"""

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from mnist_classifier import TwoLayerNN, set_seed, get_data_loaders, open_image

def run_task(seed=42, hidden_size=128, batch_size=64, learning_rate=0.001):
    """
    Run Task 5: Feature analysis using t-SNE visualization.
    
    Args:
        seed: Random seed for reproducibility
        hidden_size: Size of the hidden layer (from optimal parameters in Task 4)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary with model and visualization data
    """
    print("\n=== Task 5: Feature Analysis ===")
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data loaders
    data = get_data_loaders(batch_size=batch_size)
    train_loader = data['train_loader']
    
    # Initialize model
    print("Training model for feature analysis...")
    model = TwoLayerNN(hidden_size=hidden_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
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
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Extract features for t-SNE visualization
    print("\nExtracting features for t-SNE visualization...")
    
    # Create a subset for visualization (t-SNE is computationally intensive)
    subset_size = 2000
    
    # Create a subset of the training data for visualization
    full_train_dataset = torch.utils.data.Subset(train_loader.dataset, 
                                                indices=torch.randperm(len(train_loader.dataset))[:subset_size])
    subset_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=200, shuffle=False)
    
    # Extract features
    all_inputs = []
    all_hidden = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get flattened inputs
            inputs = images.view(-1, 28*28)
            
            # Get model outputs and hidden features
            outputs, hidden = model(images)
            
            # Store features and labels
            all_inputs.append(inputs.cpu().numpy())
            all_hidden.append(hidden.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_inputs = np.vstack(all_inputs)
    all_hidden = np.vstack(all_hidden)
    all_labels = np.concatenate(all_labels)
    
    # Apply t-SNE dimensionality reduction
    print("Applying t-SNE to input features...")
    tsne_input = TSNE(n_components=2, random_state=seed)
    input_2d = tsne_input.fit_transform(all_inputs)
    
    print("Applying t-SNE to hidden features...")
    tsne_hidden = TSNE(n_components=2, random_state=seed)
    hidden_2d = tsne_hidden.fit_transform(all_hidden)
    
    # Create plots
    print("Creating t-SNE visualizations...")
    plt.figure(figsize=(20, 10))
    
    # Plot input features
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(input_2d[:, 0], input_2d[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('t-SNE Visualization of Input Features (784-dimensional space)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Plot hidden features
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.title(f't-SNE Visualization of Hidden Features ({hidden_size}-dimensional space)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig('tsne_visualizations.png')
    plt.close()
    open_image('tsne_visualizations.png')
    
    print("t-SNE visualizations saved as 'tsne_visualizations.png'")
    
    # Analysis of feature space
    # Calculate intra-class and inter-class distances in hidden space
    distances = {}
    
    # Compute average distance between samples of the same class
    intra_class_distances = []
    for digit in range(10):
        mask = (all_labels == digit)
        if np.sum(mask) > 1:  # Need at least 2 samples
            features = hidden_2d[mask]
            # Compute pairwise distances within this class
            dists = []
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    dist = np.linalg.norm(features[i] - features[j])
                    dists.append(dist)
            if dists:
                intra_class_distances.append(np.mean(dists))
    
    # Compute average distance between different classes
    inter_class_distances = []
    for digit1 in range(10):
        for digit2 in range(digit1+1, 10):
            mask1 = (all_labels == digit1)
            mask2 = (all_labels == digit2)
            features1 = hidden_2d[mask1]
            features2 = hidden_2d[mask2]
            
            # Sample to avoid excessive computation
            sample_size = min(50, len(features1), len(features2))
            features1_sample = features1[:sample_size]
            features2_sample = features2[:sample_size]
            
            # Compute distances between classes
            dists = []
            for i in range(len(features1_sample)):
                for j in range(len(features2_sample)):
                    dist = np.linalg.norm(features1_sample[i] - features2_sample[j])
                    dists.append(dist)
            if dists:
                inter_class_distances.append(np.mean(dists))
    
    avg_intra_class = np.mean(intra_class_distances) if intra_class_distances else None
    avg_inter_class = np.mean(inter_class_distances) if inter_class_distances else None
    
    if avg_intra_class and avg_inter_class:
        separation_ratio = avg_inter_class / avg_intra_class
        print(f"\nHidden Feature Space Analysis:")
        print(f"Average Intra-class Distance: {avg_intra_class:.4f}")
        print(f"Average Inter-class Distance: {avg_inter_class:.4f}")
        print(f"Separation Ratio (Inter/Intra): {separation_ratio:.4f}")
        print("A higher ratio indicates better class separation in the hidden feature space")
    
    return {
        'model': model,
        'input_features': all_inputs,
        'hidden_features': all_hidden,
        'labels': all_labels,
        'tsne_input': input_2d,
        'tsne_hidden': hidden_2d,
        'intra_class_distances': avg_intra_class,
        'inter_class_distances': avg_inter_class
    }

if __name__ == "__main__":
    run_task()
