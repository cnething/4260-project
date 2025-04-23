# XLM-RoBERTa + LSTM Hybrid for Natural Language Inference (NLI)
# Features
Multilingual Understanding with XLM-RoBERTa.

LSTM Layer added for better sequential context understanding.

Mixed Precision Training for faster training on modern GPUs.

Early Stopping to prevent overfitting.

Layer Freezing: Freeze first N layers of RoBERTa for memory efficiency.

Confidence Scores output for predictions.

GPU memory optimization with periodic garbage collection.

# Project Structure
main(): Orchestrates loading data, training the model, and generating predictions.

Data Loading:

Reads train.csv and test.csv.

Splits train data into training and validation sets.

Model:

Hybrid of XLM-RoBERTa and a 2-layer BiLSTM.

Freezes first 6 layers of RoBERTa by default.

Training:

Optimized for T4 GPUs using mixed precision (torch.cuda.amp).

Tracks best model by validation F1 Score and Accuracy.

Logs training metrics to training_log.txt.

Inference:

Predicts test labels and saves to submission.csv and submission_with_confidence.csv.

# Requirements
Install the necessary packages:

pip install transformers datasets tqdm scikit-learn pandas numpy torch

# Usage
1. Prepare your data:

Ensure train.csv and test.csv are in the working directory.

Format:

Columns: premise, hypothesis, label (for train).

Labels: entailment, neutral, contradiction.

2. Run the script:

python xlm_roberta_lstm.py

3. Outputs:

Trained model: model_outputs/best_xlm_roberta_lstm_model.pt

Submissions: model_outputs/submission.csv and submission_with_confidence.csv

Log: model_outputs/training_log.txt

# Key Hyperparameters
TRAIN_BATCH_SIZE = 16

EVAL_BATCH_SIZE = 32

MAX_SEQ_LENGTH = 128

EPOCHS = 5

LEARNING_RATE = 2e-5

FREEZE_LAYERS = 6




