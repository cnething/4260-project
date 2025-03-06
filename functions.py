# Libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK Data Needed
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function to tokenized sentenzes in english that have already been set to lower case
# if not in lowercase change word_tokenize(sent) to word_tokenize(sent.lower())
def preprocess(sentences):
    tokenized_sentences = [word_tokenize(sent) for sent in sentences]
    filtered_sentences = [[word for word in tokens if word.isalnum() and word not in stop_words] 
                          for tokens in tokenized_sentences]
    return filtered_sentences