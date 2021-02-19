import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

def confusion(df, preds, y):
    aux_df = pd.concat([df.label, df.label.cat.codes], axis=1).rename(columns={0:'label_code'}).drop_duplicates().sort_values('label')
    conf_matrix = confusion_matrix(y, preds)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['label'].values, 
                yticklabels=aux_df['label'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()