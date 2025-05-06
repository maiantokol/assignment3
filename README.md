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

- `mnist_simple.py`: Complete implementation of all tasks in a single file
- `report_template.tex`: LaTeX template for the assignment report

## How to Run

Run the script and choose which task to execute when prompted:

```bash
python mnist_simple.py
```

You'll see a menu where you can select a specific task (1-5) or choose 'all' to run all tasks sequentially.

For example, to run Task 1, enter '1' when prompted.

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
