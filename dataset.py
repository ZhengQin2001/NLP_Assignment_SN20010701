import torch
from torch.utils.data import Dataset
import pandas as pd

class ParaphraseDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text1 = self.data.iloc[idx]['text1']
        text2 = self.data.iloc[idx]['text2']
        label = self.data.iloc[idx]['label']

        if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
            raise ValueError(f"Invalid data at index {idx}: {text1}, {text2}")

        inputs = self.tokenizer(text1, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        targets = self.tokenizer(text2, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = targets['input_ids'].squeeze(0)
        return item
