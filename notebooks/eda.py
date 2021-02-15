#%%
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
#%%
df = pd.read_csv('../data/datasets/contents.csv', dtype={'Category':'category'})
df = pd.read_csv('../data/datasets/contents.csv')
df
# %%
df.groupby('Category').count()[['Content']].plot.bar()
df.groupby('label').count()[['text']].plot.bar()
# %%
df['News length'] = df['Content'].str.len()
df.boxplot(column='News length', by='Category')
df.boxplot(column='News length', by='label')
# %%
df[df['News length'] < df['News length'].quantile(0.95)].boxplot(column='News length', by='Category')
df[df['News length'] < df['News length'].quantile(0.95)].boxplot(column='News length', by='label')
# %%
with open('../data/datasets/tfidf/features_train.pickle', 'rb') as f:
    features_train = pickle.load(f)
with open('../data/datasets/tfidf/labels_train.pickle', 'rb') as f:
    labels_train = pickle.load(f)

category_codes = {j:i for i, j in enumerate(df.Category.cat.categories)}
category_codes = {'bug':0, 'feature':1, 'question':2}

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")
# %%
