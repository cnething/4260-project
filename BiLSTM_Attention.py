'''
This file is written primarily by Nol for the purpose of developing 
and analyzing the BiLSTM model with an Attention Mechnaism.
Other contributions will be noted in in-line comments
'''
# Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import functions
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dense, Dropout, Multiply, Lambda

# NLTK Data Needed
nltk.download('punkt')
nltk.download('stopwords')

# Datasets
testDF = pd.read_csv("clean_test.csv")
trainDF = pd.read_csv("clean_train.csv")

# Convert Relevant columns to list
train_premises = trainDF['premise'].astype(str).tolist()
train_hypotheses = trainDF['hypothesis'].astype(str).tolist()
train_labels = trainDF['label'].tolist()

test_premises = testDF['premise'].astype(str).tolist()
test_hypotheses = testDF['hypothesis'].astype(str).tolist()

# Combine premise & hypotheses for tokenization
train_sentences = train_premises + train_hypotheses
test_sentences = test_premises + test_hypotheses

# Tokenization and stopword removal
stop_words = set(stopwords.words('english'))

# Based on Claudia's tokenization in main branch in project.py
train_tokenized_sent = functions.preprocess(train_sentences)
test_tokenized_sent = functions.preprocess(test_sentences)

# Build Vocabulary from Tokenized Sentences
# For each tokenized sentence counts each word in the sentence
word_counts = Counter(word for sentence in train_tokenized_sent for word in sentence)
# Creates a dictionary of the most common words, where the given word is the key and the value is how much they occured
vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common())}

# Convert premise and hypothesis into sequences
# Creates a list of list, where each list is a sequence of words found in both a given premise and the vocab
premise_sequences = [[vocab[word] for word in functions.preprocess([p])[0] if word in vocab] for p in train_premises]
# Same as premise_sequences
hypothesis_sequences = [[vocab[word] for word in functions.preprocess([h])[0] if word in vocab] for h in train_hypotheses]

# Pad Sequences
MAX_LEN = 20 # Max length a given sequence will be, adds padding to match
premise_padded = pad_sequences(premise_sequences, maxlen=MAX_LEN, padding='post')
hypothesis_padded = pad_sequences(hypothesis_sequences, maxlen=MAX_LEN, padding='post')

# Convert labels to categorical values
labels = tf.keras.utils.to_categorical(train_labels, num_classes=3)