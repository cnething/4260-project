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

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data
train = pd.read_csv("train.csv", on_bad_lines='skip', quoting=3)
test = pd.read_csv("test.csv")

# convert to Hugging Face dataset
trainHF = Dataset.from_pandas(train[['premise', 'hypothesis', 'label']])
testHF = Dataset.from_pandas(test[['premise', 'hypothesis']])

# set tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
model.to(device)

# tokenization function
def tokenize_function(text):
    return tokenizer(text["premise"], text["hypothesis"], truncation=True)

# tokenize
train = trainHF.map(tokenize_function, batched=True)
test = testHF.map(tokenize_function, batched=True)

# padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }
# training args
training_args = TrainingArguments(
    output_dir="./checkPoints", 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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