"""
Task 1: MNIST classification
Reproduce the code shown in class: load the MNIST train and test sets, 
implement a basic two-layer fully connected neural network, define the 
cross entropy loss function and optimizer, and implement train and evaluation procedures.
"""

from mnist_classifier import run_experiment

if __name__ == "__main__":
    print("Task 1: Basic MNIST Classification")
    
    # Run the basic experiment with default parameters
    results = run_experiment(
        hidden_size=128,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=10,
        seed=42,
        use_validation=False
    )
    
    print(f"Final test accuracy: {results['final_test_acc']:.2f}%")
    print(f"Final test loss: {results['final_test_loss']:.4f}")
    print("Training and test loss curves have been saved as 'loss_curves.png'")
    print("Examples of misclassified images have been saved as 'misclassified_images.png'")
