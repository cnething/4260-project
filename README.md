# Multilingual NLI Dataset Preprocessing & Analysis

This project focuses on the preprocessing, cleaning, analysis, and entity extraction of a multilingual Natural Language Inference (NLI) dataset. It leverages various tools including langdetect, transformers, nltk, and matplotlib for effective handling and visualization of data.

# Overview
This script performs:

Loading and basic exploration of train/test datasets.

Language detection for multilingual support.

Text cleaning including normalization, stopword removal, and special character handling.

Visualization of language and label distributions.

Named Entity Recognition (NER) to extract entities from the cleaned data.

# Data Description
train.csv: Contains the training set with columns like premise, hypothesis, label, and language.

test.csv: Contains the test set with similar structure but typically without labels.

Label Info:

  0: Entailment

  1: Contradiction

  2: Neutral

# Installation

pip install pandas numpy matplotlib tqdm nltk langdetect datasets transformers

# Usage
1. Place train.csv and test.csv in the same directory as the script.

2. Run the script to:

  Detect and clean the text data.

  Visualize language and class distributions.

  Extract and analyze named entities from the data.

# Features
Language Detection:
  Uses langdetect to identify the language of both premise and hypothesis texts for better multilingual processing.

Text Cleaning:
  Unicode normalization.

  Removal of special characters and stopwords.

  Handles multiple languages dynamically.

Data Analysis:
  Missing values and duplicates check.

  Class distribution and language distribution.

  Per-language class breakdown with visualizations.

Named Entity Recognition:
  Extracts named entities using Hugging Face’s dslim/bert-base-NER model.

  Analyzes top entities for insights.

# Visualization
The script provides the following visual outputs:

Language Distribution: 
Bar chart showing the percentage of each detected language.

Class Distribution:
Bar chart showing the balance of entailment, contradiction, and neutral labels.

Class Distribution per Language:
Stacked bar chart to understand class prevalence across languages.

Top 20 Named Entities:
Horizontal bar chart showing the most frequent entities extracted from text.
