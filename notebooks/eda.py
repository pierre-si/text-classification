#%%
import pandas as pd
#%%
df = pd.read_csv('../data/datasets/contents.csv', dtype={'Category':'category'})
df
# %%
df.groupby('Category').count()[['Content']].plot.bar()
df['News length'] = df['Content'].str.len()
# %%
df['News length'] = df['Content'].str.len()
df.boxplot(column='News length', by='Category')
# %%
df[df['News length'] < df['News length'].quantile(0.95)].boxplot(column='News length', by='Category')
# %%
