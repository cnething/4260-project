# XLM-RoBERTa Text Classification (Hugging Face Transformers)

This project fine-tunes the `xlm-roberta-base` multilingual transformer model to perform text classification using premise–hypothesis pairs. The model is trained and evaluated using Hugging Face's `Trainer` API.

Make sure to install the required libraries first:

!pip install datasets
!pip install evaluate
!pip install transformers

Model:

Base Model- xlm-roberta-base
Task: Multiclass sequence classification (3 classes)
Framework- Hugging Face Transformers
Tokenizer- XLMRobertaTokenizer

Current training configurations:

Batch size-	10
Learning rate-	2e-5
Epochs-	10
Evaluation strategy-	Every epoch
Early stopping-	Patience = 3
Metric for best model-	F1 (weighted)
Logging steps-	10

Metrics used:

Accuracy
F1 Score
Precision
Recall

Output:

Model is evaluated on validation set
Predictions for test set are saved to test_with_predictions.csv


https://colab.research.google.com/drive/1J2566y4UONahnFvUQ3ZYU1Nmrspjz3sQ#scrollTo=pQMLyvRtMdz5
