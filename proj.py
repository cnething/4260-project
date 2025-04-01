import pandas as pd
from datasets import Dataset
from transformers import (
    XLMRobertaTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
import evaluate
import numpy as np
import torch

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data
trainDF = pd.read_csv("train.csv", on_bad_lines='skip', quoting=3)
testDF = pd.read_csv("test.csv")

# convert to huggingface dataset
train = Dataset.from_pandas(trainDF[['premise', 'hypothesis', 'label']])
test = Dataset.from_pandas(testDF[['premise', 'hypothesis']])

# set the tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
model.to(device)

# tokenization function
def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

train = train.map(tokenize_function, batched=True)
test = test.map(tokenize_function, batched=True)

# padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }

# training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=train,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train the model
trainer.train()

# predict on test set
predictions = trainer.predict(test)
predicted_labels = np.argmax(predictions.predictions, axis=-1)

# export predictions
testDF["predicted_label"] = predicted_labels
testDF.to_csv("test_with_predictions.csv", index=False)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get predictions
preds_output = trainer.predict(train)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# initialize lists to store logs
train_epochs, train_loss = [], []
val_epochs, val_loss = [], []
val_accuracy, val_f1 = [], []

# parse the logs
for log in trainer.state.log_history:
    if "loss" in log and "epoch" in log and "eval_loss" not in log:
        train_epochs.append(log["epoch"])
        train_loss.append(log["loss"])
    if "eval_loss" in log and "epoch" in log:
        val_epochs.append(log["epoch"])
        val_loss.append(log["eval_loss"])
        val_accuracy.append(log.get("eval_accuracy", None))
        val_f1.append(log.get("eval_f1", None))

# plot the loss
plt.figure(figsize=(10, 4))
plt.plot(train_epochs, train_loss, label="Training Loss", marker='o')
plt.plot(val_epochs, val_loss, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# plot the accuracy & F1
plt.figure(figsize=(10, 4))
plt.plot(val_epochs, val_accuracy, label="Validation Accuracy", marker='o')
plt.plot(val_epochs, val_f1, label="Validation F1", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Accuracy and F1 Over Epochs")
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# binarize true labels for ROC
y_score = preds_output.predictions
y_true_bin = label_binarize(y_true, classes=[0, 1, 2]) 

# compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# plot all ROC curves
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
