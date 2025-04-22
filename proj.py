# Claudia
import torch
import pandas as pd
import numpy as np
import random
import evaluate
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# set seed the seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from google.colab import files
uploaded = files.upload()

uploaded = files.upload()

# load the data
trainDF = pd.read_csv("train.csv", on_bad_lines='skip', quoting=3)
testDF = pd.read_csv("test.csv")

print(f"Train data shape: {trainDF.shape}")
print(f"Test data shape: {testDF.shape}")
print(f"Label distribution: {trainDF['label'].value_counts(normalize=True)}")

# text cleaning
def clean_text(text):
    if isinstance(text, str):
        # remove all extra spaces
        text = ' '.join(text.split())
        return text
    return ""

trainDF['premise'] = trainDF['premise'].apply(clean_text)
trainDF['hypothesis'] = trainDF['hypothesis'].apply(clean_text)
testDF['premise'] = testDF['premise'].apply(clean_text)
testDF['hypothesis'] = testDF['hypothesis'].apply(clean_text)

# split into train and validation sets
train_df, valid_df = train_test_split(
    trainDF[['premise', 'hypothesis', 'label']],
    test_size=0.1,
    stratify=trainDF['label'],
    random_state=seed
)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")
print(f"Train label distribution: {train_df['label'].value_counts(normalize=True)}")
print(f"Validation label distribution: {valid_df['label'].value_counts(normalize=True)}")

# convert to Hugging Face datasets
train = Dataset.from_pandas(train_df.reset_index(drop=True))
valid = Dataset.from_pandas(valid_df.reset_index(drop=True))
test = Dataset.from_pandas(testDF[['premise', 'hypothesis']])

# load model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# find max sequence length
sample_texts = [(p, h) for p, h in zip(trainDF['premise'][:1000], trainDF['hypothesis'][:1000])]
lengths = [len(tokenizer.encode(p, h)) for p, h in sample_texts]
print(f"95th percentile length: {np.percentile(lengths, 95)}")
print(f"99th percentile length: {np.percentile(lengths, 99)}")
max_length = min(192, int(np.percentile(lengths, 95)))
print(f"Using max_length: {max_length}")

# tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation="longest_first",
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=False,
        return_tensors="pt"
    )

    # apply tokenization
train = train.map(tokenize_function, batched=True)
valid = valid.map(tokenize_function, batched=True)
test = test.map(tokenize_function, batched=True)

# data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load and compute metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average='macro')["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average='macro')["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average='macro')["f1"],
    }

# load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

# training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    label_smoothing_factor=0.1,
    fp16=True,
    report_to="tensorboard",
    save_total_limit=2,
)

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# train the model
print("Starting training...")
trainer.train()

# evaluate on validation set
print("Evaluating model...")
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# save best model
model_path = "./best-model"
trainer.save_model(model_path)
print(f"Best model saved to {model_path}")

# predict on test set
print("Generating predictions...")
preds = trainer.predict(test)
predicted_labels = np.argmax(preds.predictions, axis=-1)

# Save predictions
submission = pd.DataFrame({
    'id': testDF['id'],
    'label': predicted_labels
})

submission.to_csv("submission.csv", index=False)
print("Saved clean predictions to submission.csv")

testDF["predicted_label"] = predicted_labels
testDF.to_csv("test_with_predictions.csv", index=False)
print("Saved full test data with predictions to test_with_predictions.csv")

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# predictions on validation set
preds_output = trainer.predict(valid)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

# confusion Matrix
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Entailment", "Neutral", "Contradiction"])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Confusion Matrix")
plt.show()

# classification Report
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Entailment", "Neutral", "Contradiction"]))

train_epochs, train_loss = [], []
val_epochs, val_loss = [], []
val_accuracy, val_f1, val_precision, val_recall = [], [], [], []

for log in trainer.state.log_history:
    if "loss" in log and "epoch" in log and "eval_loss" not in log:
        train_epochs.append(log["epoch"])
        train_loss.append(log["loss"])
    if "eval_loss" in log and "epoch" in log:
        val_epochs.append(log["epoch"])
        val_loss.append(log["eval_loss"])
        val_accuracy.append(log.get("eval_accuracy"))
        val_f1.append(log.get("eval_f1"))
        val_precision.append(log.get("eval_precision"))
        val_recall.append(log.get("eval_recall"))

# Plot Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(train_epochs, train_loss, label="Training Loss", marker='o')
plt.plot(val_epochs, val_loss, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy, F1, Precision, Recall
plt.figure(figsize=(10, 4))
plt.plot(val_epochs, val_accuracy, label="Accuracy", marker='o')
plt.plot(val_epochs, val_f1, label="F1", marker='o')
plt.plot(val_epochs, val_precision, label="Validation Precision", marker='o')
plt.plot(val_epochs, val_recall, label="Recall", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# ROC Curve (Multiclass)
y_score = preds_output.predictions
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()