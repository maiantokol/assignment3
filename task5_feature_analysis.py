"""
Task 5: Feature analysis
Extract and visualize the hidden features of the model using t-SNE.
Compare the t-SNE visualizations of the input features and hidden features.
Analyze what the differences between these visualizations tell us about the model.
"""

from mnist_classifier import run_experiment, feature_analysis, set_seed
from torchvision import datasets, transforms
import torch

if __name__ == "__main__":
    print("Task 5: Feature Analysis")
    
    # Set seed for reproducibility
    SEED = 42
    set_seed(SEED)
    
    # Train a model with the best parameters from grid search
    # For this example, let's assume these are the best parameters:
    print("Training a model for feature analysis...")
    results = run_experiment(
        hidden_size=128,  # Can be replaced with best found in Task 4
        batch_size=64,    # Can be replaced with best found in Task 4
        learning_rate=0.001,  # Can be replaced with best found in Task 4
        num_epochs=10,
        seed=SEED,
        use_validation=False
    )
    
    # Get the trained model
    model = results['model']
    device = results['device']
    
    # Load the MNIST dataset for feature extraction
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Perform t-SNE analysis on the features
    print("\nPerforming t-SNE analysis...")
    input_features, hidden_features, labels = feature_analysis(
        model, 
        train_dataset, 
        device,
        filename_prefix="task5_tsne"
    )
    
    print("\n===== Feature Analysis Results =====")
    print(f"Number of samples used for t-SNE: {len(labels)}")
    print(f"Input feature dimension: {input_features.shape[1]}")
    print(f"Hidden feature dimension: {hidden_features.shape[1]}")
    
    print("\nt-SNE visualizations have been saved:")
    print("- Input features: task5_tsne_input_features.png")
    print("- Hidden features: task5_tsne_hidden_features.png")
    
    print("\n===== Analysis =====")
    print("""
Expected observations from the t-SNE visualizations:

1. Input Features:
   - The original MNIST digits in high-dimensional space (784 dimensions) are projected to 2D
   - Some natural clustering might be visible, but likely with significant overlap
   - Similar digits (e.g., 4 and 9, or 3 and 8) may not be well separated
   - The projection is based solely on pixel intensities

2. Hidden Features:
   - The neural network's hidden layer transforms the input into a more meaningful representation
   - We expect to see clearer separation between different digit classes
   - Similar digits should form more distinct clusters
   - The network has learned to extract features that distinguish between classes

3. Differences between the visualizations:
   - The hidden features should show better class separation than input features
   - This demonstrates the neural network's ability to transform raw inputs into more useful representations
   - The separation of clusters in hidden features indicates what the model has learned to recognize
   - Areas where digits still overlap in the hidden space suggest challenges for the model

These visualizations help us understand:
   - How the neural network transforms data through its layers
   - Which digits are harder for the network to distinguish
   - How the hidden layer organizes the feature space to facilitate classification
    """)
