#!/usr/bin/env python3
"""
Main script to run the MNIST assignment tasks.

This script provides a command-line interface to run individual tasks or all tasks in sequence.

Usage:
    python run_assignment.py --task [1-5]  # Run a specific task
    python run_assignment.py --all         # Run all tasks in sequence
"""

import argparse
import time
import sys
import importlib

# Import task modules
import task1_basic_classification
import task2_mitigate_randomness
import task3_validation_dataset
import task4_grid_search
import task5_feature_analysis

def run_task(task_number):
    """
    Run a specific task by number.
    
    Args:
        task_number (int): Task number to run (1-5)
    
    Returns:
        bool: Whether the task completed successfully
    """
    task_modules = {
        1: task1_basic_classification,
        2: task2_mitigate_randomness,
        3: task3_validation_dataset,
        4: task4_grid_search,
        5: task5_feature_analysis
    }
    
    if task_number not in task_modules:
        print(f"Error: Task {task_number} is not valid. Choose from tasks 1-5.")
        return False
    
    try:
        print(f"\n{'-'*50}")
        print(f"Running Task {task_number}")
        print(f"{'-'*50}\n")
        
        start_time = time.time()
        task_module = task_modules[task_number]
        result = task_module.run_task()
        elapsed_time = time.time() - start_time
        
        print(f"\nTask {task_number} completed successfully in {elapsed_time:.2f} seconds.")
        return True
    except Exception as e:
        print(f"\nError running task {task_number}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tasks():
    """
    Run all tasks in sequence.
    
    Returns:
        int: Number of tasks completed successfully
    """
    print("\n" + "="*50)
    print("Running all tasks in sequence")
    print("="*50 + "\n")
    
    successful_tasks = 0
    for task_number in range(1, 6):
        if run_task(task_number):
            successful_tasks += 1
    
    print(f"\nCompleted {successful_tasks} out of 5 tasks successfully.")
    return successful_tasks

def main():
    """Parse command-line arguments and run appropriate task(s)."""
    parser = argparse.ArgumentParser(description="Run MNIST assignment tasks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, choices=range(1, 6),
                       help="Run a specific task (1-5)")
    group.add_argument("--all", action="store_true",
                       help="Run all tasks in sequence")
    
    args = parser.parse_args()
    
    if args.task:
        success = run_task(args.task)
        sys.exit(0 if success else 1)
    elif args.all:
        successful_tasks = run_all_tasks()
        sys.exit(0 if successful_tasks == 5 else 1)

if __name__ == "__main__":
    main()
