from transformers import XLMRobertaTokenizer
import pandas as pd
import numpy as np
import unicodedata
import re


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

# apply cleaning function to premise and hypothesis columns
for df in [trainDF, testDF]:
    for col in ["premise", "hypothesis"]:
        df[col] = df[col].map(clean_text)

# convert text to lowercase
trainDF = trainDF.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
testDF = testDF.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def tokenize(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

trainDF["premise_tokenized"] = trainDF["premise"].apply(lambda x: tokenize(x))
trainDF["hypothesis_tokenized"] = trainDF["hypothesis"].apply(lambda x: tokenize(x))

testDF["premise_tokenized"] = testDF["premise"].apply(lambda x: tokenize(x))
testDF["hypothesis_tokenized"] = testDF["hypothesis"].apply(lambda x: tokenize(x))

print(trainDF.head())