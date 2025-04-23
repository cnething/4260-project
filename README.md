# Revel
# DeBERTa and mDeBERTa

# Features
Utilizes Hugging Face Transformers for state-of-the-art NLP.

Loads and processes datasets using datasets and pandas.

Performs training using AutoModelForSequenceClassification and the Trainer API.

Employs EarlyStoppingCallback to prevent overfitting.

Evaluates model with various metrics via evaluate library.

Visualizes data and results with matplotlib and seaborn.

# Dependencies
Install the required packages:
pip install transformers datasets evaluate torch pyarrow matplotlib seaborn

#Usage
1. Dataset:

Ensure your CSV file (e.g., train.csv) is correctly formatted and accessible in the working directory.

The notebook expects this file to contain at least:

Text Columns for inputs (e.g., premise, hypothesis).

Label Column for classification targets.

2. Running the Notebook:

Load dataset via:
data = load_dataset('csv', data_files='train.csv', split='train')

Perform tokenization with AutoTokenizer.

Fine-tune a Transformer model using Trainer.

Evaluate performance metrics such as accuracy, F1-score, etc.

3. Model:

The notebook uses AutoModelForSequenceClassification, supporting various pretrained models like BERT, RoBERTa, XLM-RoBERTa, etc.

# Key Components
Device Setup: Automatically uses GPU if available.

Training: Configured through TrainingArguments.

Evaluation: Real-time validation using evaluate metrics and classification reports.

Visualization: Metrics plotted over epochs.

# Outputs
Trained model weights (optional, based on notebook settings).

Evaluation metrics for test/validation sets.

Plots showing training progress and performance.

