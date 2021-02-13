#%%
import pickle

from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
#%%
DATA_PATH = "../data/datasets/tfidf/"
df = pd.read_csv('../data/datasets/contents.csv', dtype={'Category': 'category'})

with open(DATA_PATH+'features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)

with open(DATA_PATH+'labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)

with open(DATA_PATH+'features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)

with open(DATA_PATH+'labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)

#%%
features_train = features_train[:10_000]
labels_train = labels_train[:10_000]

# %%
mnbc = MultinomialNB()
mnbc
# %%
# FIT
mnbc.fit(features_train, labels_train)
# PREDS
mnbc_pred = mnbc.predict(features_test)
# %%
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnbc.predict(features_train)))
# %%
# Test accuracy
# 0.696
print("The test accuracy is: ")
print(accuracy_score(labels_test, mnbc_pred))
# %%
# Classification report
print("Classification report")
print(classification_report(labels_test,mnbc_pred))
# %%
aux_df = pd.DataFrame([['bug', 0], ['feature', 1], ['question', 2]], columns=['Category', 'Category_Code'])

conf_matrix = confusion_matrix(labels_test, mnbc_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=aux_df['Category'].values, 
            yticklabels=aux_df['Category'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()
# %%
d = {
     'Model': 'Multinomial Naïve Bayes',
     'Training Set Accuracy': accuracy_score(labels_train, mnbc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, mnbc_pred)
}

df_models_mnbc = pd.DataFrame(d, index=[0])
# %%
with open('../models/best_mnbc.pickle', 'wb') as output:
    pickle.dump(mnbc, output)
    
with open('../models/df_models_mnbc.pickle', 'wb') as output:
    pickle.dump(df_models_mnbc, output)
# %%
