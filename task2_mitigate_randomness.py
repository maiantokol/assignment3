"""
Task 2: Mitigate pseudorandomness
Adapt your code from Task 1 so that it produces the same results at each run.
Run your non-random code using 5 different seed numbers.
Report the mean and standard deviation of the final test errors.
Determine whether the model can be considered robust to the choice of a seed number.
"""

from mnist_classifier import run_experiment, set_seed
import numpy as np
import matplotlib.pyplot as plt

# List of seed values to test
SEED_VALUES = [42, 123, 456, 789, 1024]
NUM_EPOCHS = 10

if __name__ == "__main__":
    print("Task 2: Mitigating Pseudorandomness")
    
    all_test_losses = []
    all_test_accuracies = []
    all_loss_histories = []
    
    # Run experiment with different seeds
    for seed in SEED_VALUES:
        print(f"\nRunning experiment with seed {seed}")
        
        # Set the global seed for reproducibility
        set_seed(seed)
        
        # Run the experiment
        results = run_experiment(
            hidden_size=128,
            batch_size=64, 
            learning_rate=0.001,
            num_epochs=NUM_EPOCHS,
            seed=seed,
            use_validation=False
        )
        
        # Collect results
        all_test_losses.append(results['final_test_loss'])
        all_test_accuracies.append(results['final_test_acc'])
        all_loss_histories.append(results['test_losses'])
    
    # Calculate statistics
    mean_test_loss = np.mean(all_test_losses)
    std_test_loss = np.std(all_test_losses)
    mean_test_accuracy = np.mean(all_test_accuracies)
    std_test_accuracy = np.std(all_test_accuracies)
    
    # Print statistics
    print("\n===== Results Summary =====")
    print(f"Test losses for all seeds: {[f'{loss:.4f}' for loss in all_test_losses]}")
    print(f"Mean test loss: {mean_test_loss:.4f}")
    print(f"Standard deviation of test loss: {std_test_loss:.4f}")
    print(f"Mean test accuracy: {mean_test_accuracy:.2f}%")
    print(f"Standard deviation of test accuracy: {std_test_accuracy:.2f}%")
    
    # Plot test loss curves for all seeds
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(all_loss_histories):
        plt.plot(range(1, NUM_EPOCHS + 1), losses, label=f'Seed {SEED_VALUES[i]}')
    
    plt.title('Test Loss Curves for Different Seeds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('seed_test_loss_curves.png')
    plt.close()
    
    print("\nTest loss curves for different seeds have been saved as 'seed_test_loss_curves.png'")
    
    # Robustness analysis
    cv = std_test_loss / mean_test_loss  # coefficient of variation
    
    print("\n===== Robustness Analysis =====")
    print(f"Coefficient of variation: {cv:.4f}")
    
    if cv < 0.05:
        robustness = "highly robust"
    elif cv < 0.1:
        robustness = "robust"
    elif cv < 0.15:
        robustness = "moderately robust"
    else:
        robustness = "not robust"
    
    print(f"Based on the coefficient of variation, the model is {robustness} to the choice of seed.")
