import pandas as pd
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset(path, columns_needed):
    data = pd.read_csv(path, delimiter='\t', usecols=columns_needed.keys())
    data.rename(columns=columns_needed, inplace=True)
    data['text1'] = data['text1'].apply(clean_text)
    data['text2'] = data['text2'].apply(clean_text)
    return data

def load_datasets(paths):
    datasets = {name: load_dataset(*details) for name, details in paths.items()}
    return datasets
