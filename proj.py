
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import unicodedata
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt_tab')
import re
from deep_translator import GoogleTranslator


testDF = pd.read_csv("test.csv")
trainDF = pd.read_csv("train.csv")

# function to clean text
def clean_text(text):
    if not isinstance(text, str):  
        return text 
    
    # normalization 
    text = unicodedata.normalize("NFKC", text)
    
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # standardize quotes
    text = re.sub(r"[“”‘’«»]", '"', text)

    # remove parenthises keeping content inside
    text = re.sub(r"\((.*?)\)", r"\1", text).strip()
    
    return text

def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return text

print("Train premise")
trainDF['premise'] = trainDF['premise'].map(clean_text)
trainDF['premise'] = trainDF['premise'].apply(translate_text)
print("Train hypothesis")
trainDF['hypothesis'] = trainDF['hypothesis'].map(clean_text)
trainDF['hypothesis'] = trainDF['hypothesis'].apply(translate_text)
print("Test premise")
testDF['premise'] = testDF['premise'].map(clean_text)
testDF['premise'] = testDF['premise'].apply(translate_text)
print("Test hypothesis")
testDF['hypothesis'] = testDF['hypothesis'].map(clean_text)
testDF['hypothesis'] = testDF['hypothesis'].apply(translate_text)
# apply cleaning function to premise and hypothesis columns
# for df in [trainDF, testDF]:
#     for col in ["premise", "hypothesis"]:
#         print(f"for loop df: {df}, column: {col}")
#         df[col] = df[col].map(clean_text)
#         df[col] = df[col].map(translate_text)
        
print("\n done cleaning")
print(trainDF[0:5])
            

# convert text to lowercase
trainDF = trainDF.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
testDF = testDF.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

trainDF.to_csv("clean_train.csv", index=False)
testDF.to_csv("clean_test.csv", index=False)

# Tokenize text
data = " ".join(trainDF['premise'].dropna() + " " + trainDF['hypothesis'].dropna())
sentences = sent_tokenize(data)

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

# POS tagging
tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]

# Extract named entities
def extract_entities(tagged_sentences):
    return [
        (" ".join(word for word, tag in subtree.leaves()))
        for sentence in tagged_sentences
        for subtree in ne_chunk(sentence)
        if isinstance(subtree, Tree)
    ]

entities = extract_entities(tagged_sentences)

# Entity counts
counts = Counter(entities)

# DataFrame for entity analysis
df_entities = pd.DataFrame(counts.items(), columns=["Entity", "Frequency"])
df_entities = df_entities.sort_values(by="Frequency", ascending=False)

# Plot the top named entities
top_entities = df_entities.head(10)
plt.figure(figsize=(10, 5))
plt.bar(top_entities["Entity"], top_entities["Frequency"])
plt.title("Top Named Entities in Premise and Hypothesis Data")
plt.ylabel("Frequency")
plt.xlabel("Named Entity")
plt.xticks(rotation=45)
plt.show()