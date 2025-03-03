import pandas as pd
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from langdetect import detect
from datasets import Dataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from tqdm.notebook import tqdm
tqdm.pandas()
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# loads data
print("Loading data...")
trainDF = pd.read_csv("train.csv")
testDF = pd.read_csv("test.csv")

print(trainDF['language'].unique())

# checking for missing values
missing_values = trainDF.isnull().sum()
print("Missing Values Train Data: \n", missing_values)
missing_values = testDF.isnull().sum()
print("Missing Values Test Data: \n", missing_values)

# checking for duplicate entries
duplicate_rows = trainDF.duplicated().sum()
print("Duplicate Rows Train Data: ", duplicate_rows)
duplicate_rows = testDF.duplicated().sum()
print("Duplicate Rows Test Data: ", duplicate_rows)

# Checking the distribution of class labels
label_counts = trainDF['label'].value_counts()
print("Result Distribution for train data \n", label_counts)


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
        # remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # remove stop words
        if lang in stopwords.fileids():
            stop_words = set(stopwords.words(lang))
            words = text.split()
            text = ' '.join([word for word in words if word.lower() not in stop_words])
    else:
        # for other languages only remove non-linguistic characters
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

    # sort languages by count 
    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    
    langs = [lang for lang, _ in sorted_langs]
    counts = [count for _, count in sorted_langs]
    percentages = [count/total*100 for count in counts]
    
    # Plot the distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(langs, percentages)
    plt.title('Language Distribution')
    plt.xlabel('Languages')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

print("\nTraining data:")
display_lang_distribution(trainDF)

# plot the distribution of class labels
label_counts = trainDF['label'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(label_counts.index, label_counts.values, color=['blue', 'red', 'green'])
plt.xlabel("Label (0: Entailment, 1: Contradiction, 2: Neutral)")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.xticks(ticks=label_counts.index, labels=["Entailment (0)", "Contradiction (1)", "Neutral (2)"])
plt.show()


print("Unique Languages in Data:", trainDF['language'].unique())

# Create a stacked bar chart for class distribution per language
language_distribution = trainDF.pivot_table(index='language', columns='label', aggfunc='size', fill_value=0)
language_distribution.plot(kind='bar', stacked=True, figsize=(12, 8), colormap="viridis")
plt.xlabel("Language")
plt.ylabel("Count")
plt.title("Class Distribution per Language")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Label", labels=["Entailment (0)", "Contradiction (1)", "Neutral (2)"])
plt.show()

# Convert to Hugging Face dataset
train_dataset = Dataset.from_pandas(trainDF)
test_dataset = Dataset.from_pandas(testDF)

# Named Entity Recognition
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_entities(text, max_length=512):
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    entities = []
    for chunk in chunks:
        chunk_entities = ner_pipeline(chunk)
        entities.extend([entity['word'] for entity in chunk_entities if entity['entity_group'] != 'O'])
    return entities

print("\nExtracting named entities from a sample...")
sample_size = min(1000, len(train_dataset))
sample_texts = train_dataset['premise_clean'][:sample_size] + train_dataset['hypothesis_clean'][:sample_size]
all_entities = []
for text in tqdm(sample_texts):
    all_entities.extend(extract_entities(text))

# Count and analyze entities
entityCounts = Counter(all_entities)
df_entities = pd.DataFrame(entityCounts.items(), columns=["Entity", "Frequency"])
df_entities = df_entities.sort_values(by="Frequency", ascending=False)

print("\nTop 20 Named Entities:")
print(df_entities.head(20))

plt.figure(figsize=(12, 8))
top_20 = df_entities.head(20)
plt.barh(top_20['Entity'], top_20['Frequency'])
plt.xlabel('Frequency')
plt.ylabel('Entity')
plt.title('Top 20 Named Entities')
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()