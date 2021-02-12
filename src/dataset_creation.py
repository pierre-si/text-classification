import os

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
def create_dataset(path="data/bbc"):
    categories = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    
    data = []
    for cat in categories:
        cat_path = os.path.join(path, cat)
        files = os.listdir(cat_path)
        for file in files:
            file_path = os.path.join(cat_path, file)
            f = open(file_path, 'r', encoding="iso-8859-1")
            data.append([file, ''.join(f.readlines()), cat])
    df = pd.DataFrame(columns=['File_Name', 'Content', 'Category'], data=data)
    df.to_csv("data/datasets/contents.csv", index=False)

def preprocess_dataset(path='data/datasets/contents.csv'):
    df = pd.read_csv(path, dtype={'Category':'category'})

    # Removing \r, \n, contiguous whitespaces, 's, "
    df['Content_Parsed'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed'] = df['Content_Parsed'].str.replace("\n", " ")
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(" +", " ")
    df['Content_Parsed'] = df['Content_Parsed'].str.replace("'s", "")
    df['Content_Parsed'] = df['Content_Parsed'].str.replace('"', '')
    # To downcase
    df['Content_Parsed'] = df['Content_Parsed'].str.lower()
    # Removing punctuation
    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        df['Content_Parsed'] = df['Content_Parsed'].str.replace(punct_sign, '')
    # Lemmatization
    nltk.download('punkt')
    nltk.download('wordnet')
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
    df['Content_Parsed'] = lemmatized_text_list
    # STOP Words
    nltk.download('stopwords')
    # missing a few stopwords such as they've
    stop_words = list(stopwords.words('english'))
    stop_words.sort(key=len, reverse=True)
    # \b: Matches the empty string, but only at the beginning or end of a word. A word is defined as a sequence of word characters. Note that formally, \b is defined as the boundary between a \w and a \W character (or vice versa), or between \w and the beginning/end of the string. This means that r'\bfoo\b' matches 'foo', 'foo.', '(foo)', 'bar foo baz' but not 'foobar' or 'foo3'.\
    # r'…' → raw string (backslash are not treated as escape characters)
    # Pratique pour que les stop words matchent des mots uniquement et pas des parties de mot.
    # Attention ! \byou\b et \bre\b matchent "bla you're bla", car ' est considéré comme un indiquant une fin de mot. Si "you" ou "re" sont situés avant "you're" dans la liste, on se retrouve à la fin avec "'", d'où l'intérêt de trier la liste de stop_words.
    # Python str.replace() does not support regex, but pandas' series.str.replace does.
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed'] = df['Content_Parsed'].str.replace(regex_stopword, '')
    # supprimer éventuellement les "'", "-", 
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(" +", " ")
    df.to_csv("data/datasets/contents.csv", index=False)
    
if __name__ == '__main__':
    os.mkdir('data/datasets')
    create_dataset()
    preprocess_dataset()
