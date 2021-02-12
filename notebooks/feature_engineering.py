# %%
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

# %%
# Feature Engineering

# The next step is to create features from the raw text so we can train the machine learning models. The steps followed are:

    # Text Cleaning and Preparation: cleaning of special characters, downcasing, punctuation signs. possessive pronouns and stop words removal and lemmatization.
    # Label coding: creation of a dictionary to map each category to a code.
    # Train-test split: to test the models on unseen data.
    # Text representation: use of TF-IDF scores to represent text.
# %%
df = pd.read_csv('../data/datasets/contents.csv', dtype={'Category':'category'})
# %%
df.head()
df.loc[1]['Content']
# %%
# Removing \r, \n, "    ", 's, 
df['Content_Parsed'] = df['Content'].str.replace("\r", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("\n", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("   ", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("'s", "")
df['Content_Parsed'] = df['Content_Parsed'].str.replace('"', '')
# %%
# Regarding 3rd and 4th bullet, although it seems there is a special character, it won't affect us since it is not a real character: → c'est juste le caractère d'échappement.
text = "Mr Greenspan\'s"
text
# %%
# Downcase
df['Content_Parsed'] = df['Content_Parsed'].str.lower()
# %%
punctuation_signs = list("?:!.,;")
for punct_sign in punctuation_signs:
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(punct_sign, '')

# %%
# Lemmatization
nltk.download('punkt')
print("------------------------")
nltk.download('wordnet')
# %%
lemmatizer = WordNetLemmatizer()
nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):
    lemmatized_list = []
    text = df.loc[row]['Content_Parsed']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(lemmatizer.lemmatize(word, pos='v'))

    lemmatized_text = " ".join(lemmatized_list)

    lemmatized_text_list.append(lemmatized_text)

df['Content_Parsed_lem'] = lemmatized_text_list
# %%
# STOP Words
nltk.download('stopwords')
# %%
stop_words = list(stopwords.words('english'))
# %%
# \b: Matches the empty string, but only at the beginning or end of a word. A word is defined as a sequence of word characters. Note that formally, \b is defined as the boundary between a \w and a \W character (or vice versa), or between \w and the beginning/end of the string. This means that r'\bfoo\b' matches 'foo', 'foo.', '(foo)', 'bar foo baz' but not 'foobar' or 'foo3'.\
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_Parsed_lem'] = df['Content_Parsed_lem'].str.replace(regex_stopword, '')
# %%
df.loc[5]['Content']
# %%
df.loc[5]['Content_Parsed_lem']
# %%
