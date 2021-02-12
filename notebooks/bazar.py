# %%
import os

import pandas as pd
# %%
path = "../data/bbc-fulltext/bbc"
# %%
categories = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
# %%
summary_category_data = []
for cat in categories:
    files_count = len(os.listdir(os.path.join(path, cat)))
    summary_category_data.append([cat, files_count])

summary_category = pd.DataFrame(columns=['Category', 'Number_of_files'], data=summary_category_data)
summary_category
# %%
data = []
for cat in categories:
    cat_path = os.path.join(path, cat)
    files = os.listdir(cat_path)
    for file in files:
        file_path = os.path.join(cat_path, file)
        f = open(file_path, 'r', encoding="iso-8859-1")
        data.append([file, ''.join(f.readlines()), cat])
df = pd.DataFrame(columns=['File_Name', 'Content', 'Category'], data=data)
# %%
df.to_csv("../data/datasets/contents.csv")

#%%
