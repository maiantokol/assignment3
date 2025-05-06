"""
Task 3: Validation dataset
Split the MNIST dataset to three sets (instead of two): train, test, and validation.
Repeat the training required in Task 2 using five different models.
Report the test error that corresponds to the minimum validation error obtained during training.
"""

from mnist_classifier import run_experiment, set_seed
import numpy as np
import matplotlib.pyplot as plt

# List of seed values to test
SEED_VALUES = [42, 123, 456, 789, 1024]
NUM_EPOCHS = 10
VALIDATION_SIZE = 10000  # Standard size used for validation set

if __name__ == "__main__":
    print("Task 3: Using Validation Dataset")
    
    all_best_val_losses = []
    all_best_test_losses = []
    all_test_accuracies = []
    
    # Run experiment with different seeds
    for seed in SEED_VALUES:
        print(f"\nRunning experiment with seed {seed}")
        
        # Set the global seed for reproducibility
        set_seed(seed)
        
        # Run the experiment with validation set
        results = run_experiment(
            hidden_size=128,
            batch_size=64, 
            learning_rate=0.001,
            num_epochs=NUM_EPOCHS,
            seed=seed,
            use_validation=True,
            validation_size=VALIDATION_SIZE
        )
        
        # Collect results
        all_best_val_losses.append(results['best_val_loss'])
        all_best_test_losses.append(results['best_test_loss'])
        all_test_accuracies.append(results['final_test_acc'])
    
    # Calculate statistics
    mean_best_val_loss = np.mean(all_best_val_losses)
    std_best_val_loss = np.std(all_best_val_losses)
    mean_best_test_loss = np.mean(all_best_test_losses)
    std_best_test_loss = np.std(all_best_test_losses)
    
    # Print statistics
    print("\n===== Results Summary =====")
    print("For each seed, reporting the test error that corresponds to the minimum validation error:")
    print(f"{'Seed':<10} {'Best Val Loss':<15} {'Corresponding Test Loss':<25}")
    for i, seed in enumerate(SEED_VALUES):
        print(f"{seed:<10} {all_best_val_losses[i]:.4f}{' '*10} {all_best_test_losses[i]:.4f}")
    
    print("\n===== Overall Statistics =====")
    print(f"Mean best validation loss: {mean_best_val_loss:.4f}")
    print(f"Std of best validation loss: {std_best_val_loss:.4f}")
    print(f"Mean test loss at best validation: {mean_best_test_loss:.4f}")
    print(f"Std of test loss at best validation: {std_best_test_loss:.4f}")
    
    # Create a bar chart comparing best validation loss and corresponding test loss
    plt.figure(figsize=(12, 6))
    x = np.arange(len(SEED_VALUES))
    width = 0.35
    
    plt.bar(x - width/2, all_best_val_losses, width, label='Best Validation Loss')
    plt.bar(x + width/2, all_best_test_losses, width, label='Corresponding Test Loss')
    
    plt.xlabel('Seed')
    plt.ylabel('Loss')
    plt.title('Best Validation Loss and Corresponding Test Loss for Different Seeds')
    plt.xticks(x, SEED_VALUES)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('validation_test_loss_comparison.png')
    plt.close()
    
    print("\nComparison of validation and test losses has been saved as 'validation_test_loss_comparison.png'")
