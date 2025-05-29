"""
Task 2: Mitigating Pseudorandomness

This script tests the model's robustness across different random initializations
by training with multiple seeds and comparing the results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist_classifier import set_seed, open_image
from task1_basic_classification import run_task

def run_task():
    """
    Run Task 2: Training the model with different random seeds to test robustness.
    
    Returns:
        Statistics of the experiments across different seeds
    """
    print("\n=== Task 2: Mitigating Pseudorandomness ===")
    
    # Define seeds to test
    seeds = [42, 123, 456, 789, 1024]
    all_test_losses = []
    
    # Create figure for plotting test losses
    plt.figure(figsize=(10, 6))
    
    # Run training with each seed
    for i, seed in enumerate(seeds):
        print(f"\nRunning experiment with seed {seed}")
        
        # Run Task 1 with this seed
        _, _, test_losses = run_task(seed)
        
        # Store final test loss for statistics
        all_test_losses.append(test_losses[-1])
        
        # Plot test loss curve for this seed
        epochs = range(1, len(test_losses) + 1)
        plt.plot(epochs, test_losses, label=f'Seed {seed}')
    
    # Compute statistics
    mean_loss = np.mean(all_test_losses)
    std_loss = np.std(all_test_losses)
    cv = std_loss / mean_loss  # Coefficient of variation
    
    # Finalize and save plot
    plt.title('Test Loss Curves for Different Seeds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('seed_test_loss_curves.png')
    plt.close()
    open_image('seed_test_loss_curves.png')
    
    # Print statistics
    print("\n=== Seed Test Results ===")
    print(f"Seeds: {seeds}")
    print(f"Final test losses: {[f'{loss:.4f}' for loss in all_test_losses]}")
    print(f"Mean test loss: {mean_loss:.4f}")
    print(f"Standard deviation: {std_loss:.4f}")
    print(f"Coefficient of variation: {cv:.4f}")
    
    return {
        'seeds': seeds,
        'test_losses': all_test_losses,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'cv': cv
    }

if __name__ == "__main__":
    run_task()
