"""
Assignment 3: Feedforward Neural Networks using PyTorch
Main script to run all tasks
"""

import argparse
import time
import sys
from datetime import datetime

def print_header(task_number, task_name):
    print("\n" + "="*80)
    print(f"TASK {task_number}: {task_name}")
    print("="*80 + "\n")

def run_task(task_module, task_number, task_name):
    print_header(task_number, task_name)
    print(f"Starting Task {task_number} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Import and run the task module
        __import__(task_module)
        
        # Calculate and print execution time
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nTask {task_number} completed in {duration:.2f} seconds")
        return True
    except Exception as e:
        print(f"\nError running Task {task_number}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Assignment 3 tasks")
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4, 5], 
                        help="Run a specific task (1-5)")
    parser.add_argument('--all', action='store_true', 
                        help="Run all tasks sequentially")
    args = parser.parse_args()
    
    # Define tasks
    tasks = {
        1: ("task1_basic_classification", "Basic MNIST Classification"),
        2: ("task2_mitigate_randomness", "Mitigating Pseudorandomness"),
        3: ("task3_validation_dataset", "Using Validation Dataset"),
        4: ("task4_grid_search", "Grid Search for Hyperparameters"),
        5: ("task5_feature_analysis", "Feature Analysis with t-SNE")
    }
    
    # Run tasks
    if args.all:
        print("Running all tasks sequentially")
        for task_num in tasks:
            success = run_task(tasks[task_num][0], task_num, tasks[task_num][1])
            if not success and task_num < 5:  # If a task fails, skip subsequent ones
                print("Stopping due to error")
                break
    elif args.task:
        task_num = args.task
        run_task(tasks[task_num][0], task_num, tasks[task_num][1])
    else:
        print("Please specify either --task N to run a specific task, or --all to run all tasks.")
        print("Example: python run_assignment.py --task 1")
        print("Example: python run_assignment.py --all")

if __name__ == "__main__":
    print("Assignment 3: Feedforward Neural Networks using PyTorch")
    main()
