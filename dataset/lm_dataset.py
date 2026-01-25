import json
from random import sample

from torch.utils.data import Dataset
import torch
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class PretrainingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                sample = json.loads(line.strip())
                samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        encode = self.tokenizer(
            str(sample['text']),
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

        input_ids = encode['input_ids'].squeeze()

        loss_mask = input_ids != self.tokenizer.pad_token_id

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        
        loss_mask = torch.tensor(loss_mask[:-1], dtype=torch.bool)

        return X, Y, loss_mask