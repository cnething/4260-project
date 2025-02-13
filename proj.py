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


print(trainDF.head())