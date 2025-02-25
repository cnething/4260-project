import pandas as pd
import numpy as np
import unicodedata
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

# apply cleaning function to premise and hypothesis columns
# Removed trainDF as it shouldn't need to be cleaned - Nol
print("Translating Premise")
trainDF['premise'] = trainDF['premise'].apply(translate_text)
print("Cleaning Premise")
trainDF['premise'] = trainDF['premise'].apply(clean_text)
print("translating hypothesis")
trainDF['hypothesis'] = trainDF['hypothesis'].apply(translate_text)
print("Cleaning hypothesis")
trainDF['hypothesis'] = trainDF['hypothesis'].apply(clean_text)
# for df in [trainDF]:
#     for col in ["premise", "hypothesis"]:
#         print(f"for loop df: {df}, column: {col}")
#         df[col] = df[col].map(clean_text)
#         df[col] = df[col].apply(translate_text)
        
print("\n done cleaning")
print(trainDF[0:5])
            

# convert text to lowercase
trainDF = trainDF.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
testDF = testDF.apply(lambda x: x.str.lower() if x.dtype == "object" else x)


print(trainDF.head())
trainDF.to_csv("clean_train.csv", index=False)