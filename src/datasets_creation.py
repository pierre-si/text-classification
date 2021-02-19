import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
def merge_bbc_data():
    path="data/bbc"
    categories = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    
    data = []
    for cat in categories:
        cat_path = os.path.join(path, cat)
        files = os.listdir(cat_path)
        for file in files:
            file_path = os.path.join(cat_path, file)
            f = open(file_path, 'r', encoding="iso-8859-1")
            data.append([file, ''.join(f.readlines()), cat])
    df = pd.DataFrame(columns=['filename', 'text', 'label'], data=data)
    df.to_csv("datasets/bbc/contents.csv", index=False)

# train & train_extra: title, body, label
# test: title, body
def retrieve_github_data():
    train = pd.read_json('data/github/embold_train.json')
    #train_extra = pd.read_json('data/github/embold_train_extra.json')
    #train = pd.concat([train, train_extra], ignore_index=True)
    #test = pd.read_json('data/github/embold_test.json')
    # 0: bug, 1: feature, 2: question
    train['text'] = train.title + ' ' + train.body
    #test['text'] = test.title + ' ' + test.body

    df = train.drop(['title', 'body'], axis=1)
    category_codes = {0:'bug', 1:'feature', 2:'question'}
    df.label = df.label.map(category_codes)
    df.to_csv('datasets/github/contents.csv', index=False)

def retrieve_twitter_data():
    train = pd.read_csv('data/twitter/train.csv', dtype={'sentiment':'category'})
    test = pd.read_csv('data/twitter/test.csv', dtype={'sentiment':'category'})

    train = train.drop(['textID', 'selected_text'], axis=1)
    train = train.dropna().reset_index(drop=True)
    test = test.drop(['textID'], axis=1)
    train.rename(columns={'sentiment':'label'}, inplace=True)
    test.rename(columns={'sentiment':'label'}, inplace=True)

    train.to_csv('datasets/twitter/contents.csv', index=False)
    test.to_csv('datasets/twitter/test.csv', index=False)

def preprocess_text(path):
    df = pd.read_csv(path, dtype={'label':'category'})

    # Removing \r, \n, contiguous whitespaces, 's, "
    df['parsed'] = df['text'].str.replace("\r", " ")
    df['parsed'] = df['parsed'].str.replace("\n", " ")
    df['parsed'] = df['parsed'].str.replace(" +", " ")
    df['parsed'] = df['parsed'].str.replace("'s", "")
    df['parsed'] = df['parsed'].str.replace('"', '')
    # To downcase
    df['parsed'] = df['parsed'].str.lower()
    # Removing punctuation
    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        df['parsed'] = df['parsed'].str.replace(punct_sign, '')
    # Lemmatization
    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []

    for row in range(0, nrows):
        lemmatized_list = []
        text = df.loc[row]['parsed']
        text_words = text.split(" ")
        for word in text_words:
            lemmatized_list.append(lemmatizer.lemmatize(word, pos='v'))

        lemmatized_text = " ".join(lemmatized_list)
        lemmatized_text_list.append(lemmatized_text)
    df['parsed'] = lemmatized_text_list
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
        df['parsed'] = df['parsed'].str.replace(regex_stopword, '')
    # supprimer éventuellement les "'", "-", 
    df['parsed'] = df['parsed'].str.replace(" +", " ")

    # removing rows with empty string (or else they will be loaded as NaN)
    df = df[df['parsed'] != '']

    df.to_csv(path, index=False)

def split_data(folder, test_size=.1, random_state=10):
    df = pd.read_csv(folder+'contents.csv', dtype={'label':'category'})
    X_train, X_test, y_train, y_test = train_test_split(df['parsed'], df['label'].cat.codes, test_size=test_size, random_state=random_state, stratify=df['label'].cat.codes)
    with open(folder+'/X_train.pickle', 'wb') as output:
        pickle.dump(X_train, output)
    with open(folder+'/X_test.pickle', 'wb') as output:
        pickle.dump(X_test, output)
    with open(folder+'/y_train.pickle', 'wb') as output:
        pickle.dump(y_train, output)
    with open(folder+'/y_test.pickle', 'wb') as output:
        pickle.dump(y_test, output)

def generate_tfidf(folder):
    ngram_range = (1, 2) # unigram et bigram
    min_df = 10 # ignores terms with df lower than 10 (int)
    max_df = 1. # ignores terms with df larger than 100% (float)
    max_features = 300

    tfidf = TfidfVectorizer(
        encoding="utf-8",
        ngram_range=ngram_range,
        stop_words=None,
        lowercase=False,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        norm='l2',
        sublinear_tf=True) # → replaces tf with 1+log(tf)

    with open(folder+"/X_train.pickle", 'rb') as f:
        X_train = pickle.load(f)
    with open(folder+"/X_test.pickle", 'rb') as f:
        X_test = pickle.load(f)
    with open(folder+"/y_train.pickle", 'rb') as f:
        y_train = pickle.load(f)
    with open(folder+"/y_test.pickle", 'rb') as f:
        y_test = pickle.load(f)

    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test
    with open(folder+'/tfidf/features_train.pickle', 'wb') as output:
        pickle.dump(features_train, output)
    with open(folder+'/tfidf/labels_train.pickle', 'wb') as output:
        pickle.dump(labels_train, output)
    with open(folder+'/tfidf/features_test.pickle', 'wb') as output:
        pickle.dump(features_test, output)
    with open(folder+'/tfidf/labels_test.pickle', 'wb') as output:
        pickle.dump(labels_test, output)


if __name__ == '__main__':
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    # bbc
    # if not os.path.exists('datasets/bbc'):
    #     os.mkdir('datasets/bbc')
    # if not os.path.exists('datasets/bbc/tfidf'):
    #     os.mkdir('datasets/bbc/tfidf')
    # merge_bbc_data()
    # preprocess_text('datasets/bbc/contents.csv')
    # split_data('datasets/bbc/', test_size=.15)
    # generate_tfidf('datasets/bbc/')

    # github
    # if not os.path.exists('datasets/github'):
    #     os.mkdir('datasets/github')
    # if not os.path.exists('datasets/github/tfidf'):
    #     os.mkdir('datasets/github/tfidf')
    # retrieve_github_data()
    # preprocess_text('datasets/github/contents.csv')
    # split_data('datasets/github/')
    # generate_tfidf('datasets/github/')

    # twitter
    if not os.path.exists('datasets/twitter'):
        os.mkdir('datasets/twitter')
    if not os.path.exists('datasets/twitter/tfidf'):
        os.mkdir('datasets/twitter/tfidf')
    retrieve_twitter_data()
    preprocess_text('datasets/twitter/contents.csv')
    split_data('datasets/twitter/')
    generate_tfidf('datasets/twitter/')
