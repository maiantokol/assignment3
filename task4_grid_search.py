"""
Task 4: Grid search
Perform grid search over hyperparameters:
- hidden size of the intermediate layer
- batch size
- learning rate
Train models for each combination of parameters and report the test error 
corresponding to the best validation error for each combination.
"""

from mnist_classifier import run_experiment, set_seed
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns

# Define hyperparameter values to search
HIDDEN_SIZES = [64, 128, 256]
BATCH_SIZES = [32, 128]
LEARNING_RATES = [0.01, 0.001, 0.0001]
VALIDATION_SIZE = 10000
NUM_EPOCHS = 15
SEED = 42

if __name__ == "__main__":
    print("Task 4: Grid Search")
    
    # Set a fixed seed for reproducibility across all experiments
    set_seed(SEED)
    
    # Create arrays to store results
    results = []
    
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(HIDDEN_SIZES, BATCH_SIZES, LEARNING_RATES))
    total_combinations = len(param_combinations)
    
    print(f"Starting grid search with {total_combinations} hyperparameter combinations")
    
    # Perform grid search
    for i, (hidden_size, batch_size, lr) in enumerate(param_combinations):
        print(f"\nCombination {i+1}/{total_combinations}:")
        print(f"hidden_size={hidden_size}, batch_size={batch_size}, learning_rate={lr}")
        
        # Run experiment
        result = run_experiment(
            hidden_size=hidden_size,
            batch_size=batch_size,
            learning_rate=lr,
            num_epochs=NUM_EPOCHS,
            seed=SEED,
            use_validation=True,
            validation_size=VALIDATION_SIZE
        )
        
        # Store results
        results.append({
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'learning_rate': lr,
            'val_loss': result['best_val_loss'],
            'test_loss': result['best_test_loss'],
            'test_acc': result['final_test_acc']
        })
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Sort by validation loss (ascending)
    sorted_df = results_df.sort_values(by='val_loss')
    
    # Save results to CSV
    sorted_df.to_csv('grid_search_results.csv', index=False)
    
    # Find the best combination
    best_combination = sorted_df.iloc[0]
    
    # Print summary table
    print("\n===== Grid Search Results =====")
    print(f"{'Hidden Size':<15} {'Batch Size':<15} {'Learning Rate':<15} {'Val Loss':<15} {'Test Loss':<15} {'Test Acc':<15}")
    
    for _, row in sorted_df.iterrows():
        print(f"{int(row['hidden_size']):<15} {int(row['batch_size']):<15} {row['learning_rate']:<15.4f} "
              f"{row['val_loss']:<15.4f} {row['test_loss']:<15.4f} {row['test_acc']:<15.2f}%")
    
    # Print the best combination
    print("\n===== Best Combination =====")
    print(f"Hidden Size: {int(best_combination['hidden_size'])}")
    print(f"Batch Size: {int(best_combination['batch_size'])}")
    print(f"Learning Rate: {best_combination['learning_rate']:.4f}")
    print(f"Validation Loss: {best_combination['val_loss']:.4f}")
    print(f"Test Loss: {best_combination['test_loss']:.4f}")
    print(f"Test Accuracy: {best_combination['test_acc']:.2f}%")
    
    # Create a heatmap visualization for test loss
    pivot_table = sorted_df.pivot_table(
        index='batch_size', 
        columns=['hidden_size', 'learning_rate'],
        values='test_loss'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title('Test Loss for Different Hyperparameter Combinations')
    plt.tight_layout()
    plt.savefig('grid_search_heatmap.png')
    plt.close()
    
    print("\nGrid search results have been saved to 'grid_search_results.csv'")
    print("Heatmap visualization has been saved to 'grid_search_heatmap.png'")
