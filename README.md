# Assignment 3: Feedforward Neural Networks using PyTorch

This repository contains the implementation of various feedforward neural networks for the MNIST digit classification task, as required for Assignment 3.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- numpy
- pandas
- seaborn

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install the required packages:
   ```bash
   pip install torch torchvision matplotlib scikit-learn numpy pandas seaborn
   ```

## Project Structure

- `mnist_classifier.py`: Core implementation of the neural network model, training and evaluation functions
- `task1_basic_classification.py`: Basic MNIST classification (Task 1)
- `task2_mitigate_randomness.py`: Testing model robustness across different seeds (Task 2)
- `task3_validation_dataset.py`: Using validation set for model selection (Task 3)
- `task4_grid_search.py`: Grid search for hyperparameter optimization (Task 4)
- `task5_feature_analysis.py`: t-SNE visualization of features (Task 5)
- `run_assignment.py`: Main script to run individual or all tasks
- `report_template.tex`: LaTeX template for the assignment report

## How to Run

### Running Individual Tasks

To run a specific task, use:

```bash
python run_assignment.py --task [1-5]
```

For example, to run Task 1:

```bash
python run_assignment.py --task 1
```

### Running All Tasks in Sequence

To run all tasks:

```bash
python run_assignment.py --all
```

### Task Descriptions

1. **Task 1**: Implementation of a basic two-layer neural network for MNIST classification
2. **Task 2**: Testing the model's robustness across different random seeds
3. **Task 3**: Using a validation dataset for early stopping and model selection
4. **Task 4**: Performing grid search over hyperparameters
5. **Task 5**: Analyzing learned features using t-SNE visualization

## Generating the Report

1. Run all tasks to generate the necessary plots and results
2. Fill in the provided `report_template.tex` with your results and analysis
3. Compile the LaTeX document to generate the final PDF report

You can use an online LaTeX editor like Overleaf, or compile locally if you have LaTeX installed:

```bash
pdflatex report_template.tex
```

## Submission Guidelines

Your submission should include:
1. `report.pdf`: The final report with all results and analysis
2. `code.zip`: A compressed file containing all your code

## Performance Notes

- Tasks 2-5 can be computationally intensive. Consider running them on a machine with adequate resources.
- If you have limited computational resources, you can modify the MNIST dataset size as mentioned in the assignment instructions (reduce by a factor of 10).
- The grid search in Task 4 will train multiple models and may take considerable time.
