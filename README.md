# Claudia
# XLM-RoBERTa Text Classification (Hugging Face Transformers)

This project fine-tunes the `xlm-roberta-base` multilingual transformer model to perform text classification using premise–hypothesis pairs. The model is trained and evaluated using Hugging Face's `Trainer` API.

Make sure to install the required libraries first:

!pip install datasets
!pip install evaluate
!pip install transformers

# Environment
Platform: Google Colab
Language: Python 3.11+
Hardware: GPU acceleration

# Model:
Base Model- xlm-roberta-base
Task: Multiclass sequence classification (3 classes)
Framework- Hugging Face Transformers
Tokenizer- XLMRobertaTokenizer

# Usage
1. Open the notebook/script in Google Colab.

2. Upload or mount your dataset files (train.csv, test.csv).

3. Run the script xlm_roberta_final.py or notebook cells step-by-step.

4. Training, validation, and evaluation are automatically handled.

5. Outputs include:
Validation metrics over epochs
ROC curves
Confusion matrices
Loss curves
CSV files for predictions and evaluation metrics

# Model Performance
Accuracy: ~70% on validation data.

F1-Score: ~70%, balanced across classes.

ROC-AUC:
Entailment: 0.88
Neutral: 0.88
Contradiction: 0.81

# Visualizations
1. Validation Metrics Over Epochs: Accuracy, Precision, Recall, F1.

2. Multiclass ROC Curve: Per-class AUC analysis.

3. Confusion Matrix: Normalized matrix showing true vs predicted labels.

4. Training vs Validation Loss: Over epochs to assess overfitting.

# Collab Link
https://colab.research.google.com/drive/1zfhn90gj5A_nOD2XQY0DZu_nfkUJpCBz?usp=sharing
