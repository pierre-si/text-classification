import os
import pandas as pd

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
    df.to_csv("data/datasets/contents.csv")

if __name__ == '__main__':
    create_dataset()