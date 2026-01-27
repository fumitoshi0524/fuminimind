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


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                samples.append(json.loads(line.strip()))
        return samples

    def __len__(self):
        return len(self.samples)

    def _build_prompt_response(self, sample):
        if "prompt" in sample and "response" in sample:
            return sample["prompt"], sample["response"]
        if "instruction" in sample and "output" in sample:
            return sample["instruction"], sample["output"]
        if "messages" in sample and isinstance(sample["messages"], list):
            messages = sample["messages"]
            if len(messages) >= 1 and messages[-1].get("role") == "assistant":
                response = messages[-1].get("content", "")
                prompt_messages = messages[:-1]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    prompt = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    return prompt, response
        text = str(sample.get("text", ""))
        return text, ""

    def _encode_sample(self, sample):
        prompt, response = self._build_prompt_response(sample)

        if response:
            full_text = f"{prompt}{response}{self.tokenizer.eos_token or ''}"
            prompt_ids = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length
            )["input_ids"]
        else:
            full_text = prompt
            prompt_ids = []

        encode = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encode['input_ids'].squeeze()
        labels = input_ids.clone()

        prompt_len = min(len(prompt_ids), self.max_length)
        if prompt_len > 0:
            labels[:prompt_len] = -100

        labels[input_ids == self.tokenizer.pad_token_id] = -100
        valid_count = (labels != -100).sum().item()
        return input_ids, labels, valid_count

    def __getitem__(self, index):
        tries = 0
        while tries < 8:
            sample = self.samples[index]
            input_ids, labels, valid_count = self._encode_sample(sample)
            if valid_count > 0:
                return input_ids, labels
            index = (index + 1) % len(self.samples)
            tries += 1

        return input_ids, labels