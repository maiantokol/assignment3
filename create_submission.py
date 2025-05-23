#!/usr/bin/env python3
"""
Script to create submission files for Assignment 3
"""

import os
import zipfile
import argparse

def create_code_zip():
    """Create the code.zip file for submission"""
    
    # Files to include in code.zip
    code_files = [
        'mnist_classifier.py',
        'task1_basic_classification.py',
        'task2_mitigate_randomness.py',
        'task3_validation_dataset.py',
        'task4_grid_search.py',
        'task5_feature_analysis.py',
        'run_assignment.py',
        'README.md'
    ]
    
    # Create code.zip
    with zipfile.ZipFile('code.zip', 'w') as zipf:
        for file in code_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file} to code.zip")
            else:
                print(f"Warning: {file} not found, skipping")
    
    print("\nSubmission file 'code.zip' created successfully!")

if __name__ == "__main__":
    create_code_zip()
    print("\nRemember to compile your report.pdf from report_template.tex before submission!")
