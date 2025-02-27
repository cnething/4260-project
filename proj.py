import pandas as pd
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from langdetect import detect
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from tqdm import tqdm_notebook
tqdm_notebook().pandas()

# loads data
print("Loading data...")
trainDF = pd.read_csv("train.csv")
testDF = pd.read_csv("test.csv")

# function that detects languages
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

# function to clean text
def clean_text(text, lang='unknown'):
    if not isinstance(text, str):
        return str(text)
    
    # normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # language-specific cleaning
    if lang in ['en', 'es', 'fr', 'de', 'it']:  # we can add more here -Claudia
        # Remove special characters, keeping basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    else:
        # for other languages, only remove clearly non-linguistic characters
        text = re.sub(r'[^\w\s.,!?]', '', text)
    
    return text

# function to process DataFrame
def process_dataframe(df):
    print("Detecting languages...")
    df['premise_lang'] = df['premise'].progress_apply(detect_language)
    df['hypothesis_lang'] = df['hypothesis'].progress_apply(detect_language)
    
    print("Cleaning text...")
    def safe_clean_text(row):
        try:
            premise_clean = clean_text(row['premise'], row['premise_lang'])
            hypothesis_clean = clean_text(row['hypothesis'], row['hypothesis_lang'])
            return pd.Series([premise_clean, hypothesis_clean])
        except Exception as e:
            print(f"Error processing row: {e}")
            print(f"Premise: {row['premise'][:50]}...")
            print(f"Hypothesis: {row['hypothesis'][:50]}...")
            return pd.Series([row['premise'], row['hypothesis']])

    cleaned = df.progress_apply(safe_clean_text, axis=1)
    df['premise_clean'], df['hypothesis_clean'] = cleaned[0], cleaned[1]
    
    return df

# process DataFrames
print("Processing training data...")
try:
    trainDF = process_dataframe(trainDF)
    print("Training data processed successfully.")
except Exception as e:
    print(f"Error processing training data: {e}")

print("Processing test data...")
try:
    testDF = process_dataframe(testDF)
    print("Test data processed successfully.")
except Exception as e:
    print(f"Error processing test data: {e}")


# display language distribution
def display_lang_distribution(df):
    lang_counts = defaultdict(int)
    for lang in df['premise_lang']:
        lang_counts[lang] += 1
    for lang in df['hypothesis_lang']:
        lang_counts[lang] += 1
    
    total = sum(lang_counts.values())
    print("Language distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{lang}: {count} ({count/total*100:.2f}%)")

print("\nTraining data:")
display_lang_distribution(trainDF)
print("\nTest data:")
display_lang_distribution(testDF)

# convert to hugging face dataset
train_dataset = Dataset.from_pandas(trainDF)
test_dataset = Dataset.from_pandas(testDF)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)

# tokenization function
def tokenize_function(examples):
    return tokenizer(examples["premise_clean"], examples["hypothesis_clean"], truncation=True, padding="max_length")

# apply tokenization
print("\nTokenizing training data...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=4)
print("Tokenizing test data...")
tokenized_test = test_dataset.map(tokenize_function, batched=True, num_proc=4)

# Named Entity Recognition (using tokenized data)
from transformers import pipeline

ner_pipeline = pipeline("ner", model="bert-base-multilingual-cased", tokenizer=tokenizer)

def extract_entities(text, max_length=512):
    # Split long text into chunks to avoid exceeding max token limit
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    entities = []
    for chunk in chunks:
        entities.extend(ner_pipeline(chunk))
    return [entity['word'] for entity in entities]

# extracts entities from a sample of the data
print("\nExtracting named entities from a sample...")
sample_size = min(1000, len(tokenized_train))  # Adjust sample size as needed
sample_texts = tokenized_train['premise_clean'][:sample_size] + tokenized_train['hypothesis_clean'][:sample_size]
all_entities = []
for text in tqdm(sample_texts):
    all_entities.extend(extract_entities(text))

# count and analyze entities
entity_counts = Counter(all_entities)
df_entities = pd.DataFrame(entity_counts.items(), columns=["Entity", "Frequency"])
df_entities = df_entities.sort_values(by="Frequency", ascending=False)

# plot top named entities
top_entities = df_entities.head(10)
plt.figure(figsize=(12, 6))
plt.bar(top_entities["Entity"], top_entities["Frequency"])
plt.title("Top Named Entities in Sample Data")
plt.ylabel("Frequency")
plt.xlabel("Named Entity")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# prints top 20 entities and their frequencies
print("\nTop 20 Named Entities:")
print(df_entities.head(20))

# Save the processed datasets
tokenized_train.save_to_disk("processed_train")
tokenized_test.save_to_disk("processed_test")
print("\nProcessed datasets saved to 'processed_train' and 'processed_test'")
