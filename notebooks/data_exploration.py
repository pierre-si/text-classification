#%%
import pandas as pd
#%%
df = pd.read_csv('../data/datasets/contents.csv', dtype={'Category':'category'})
df
#%%
summary_category_data = []
    for cat in categories:
        files_count = len(os.listdir(os.path.join(path, cat)))
        summary_category_data.append([cat, files_count])

    summary_category = pd.DataFrame(columns=['Category', 'Number_of_files'], data=summary_category_data)
    summary_category
#%%
df.Category.cat.codes
# %%
df.groupby('Category').count()[['Content']].plot.bar()
df['News length'] = df['Content'].str.len()
# %%
df['News length'] = df['Content'].str.len()
df.boxplot(column='News length', by='Category')
# %%
df[df['News length'] < df['News length'].quantile(0.95)].boxplot(column='News length', by='Category')
# %%
