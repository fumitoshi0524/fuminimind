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
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
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
        if "conversations" in sample and isinstance(sample["conversations"], list):
            messages = sample["conversations"]
            if len(messages) >= 1 and messages[-1].get("role") == "assistant":
                response = messages[-1].get("content", "")
                prompt_messages = messages[:-1]
                prompt = None
                if hasattr(self.tokenizer, "apply_chat_template"):
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except ValueError:
                        prompt = None
                if prompt is None:
                    prompt = "".join(
                        f"{m.get('role','user')}: {m.get('content','')}\n"
                        for m in prompt_messages
                    )
                return prompt, response
        if "prompt" in sample and "response" in sample:
            return sample["prompt"], sample["response"]
        if "instruction" in sample and "output" in sample:
            return sample["instruction"], sample["output"]
        if "messages" in sample and isinstance(sample["messages"], list):
            messages = sample["messages"]
            if len(messages) >= 1 and messages[-1].get("role") == "assistant":
                response = messages[-1].get("content", "")
                prompt_messages = messages[:-1]
                prompt = None
                if hasattr(self.tokenizer, "apply_chat_template"):
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except ValueError:
                        prompt = None
                if prompt is None:
                    prompt = "".join(
                        f"{m.get('role','user')}: {m.get('content','')}\n"
                        for m in prompt_messages
                    )
                return prompt, response
        text = str(sample.get("text", ""))
        return text, ""

    def _get_eos_text(self) -> str:
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != getattr(self.tokenizer, "unk_token_id", None):
            return "<|im_end|>"
        return self.tokenizer.eos_token or ""

    def _encode_sample(self, sample):
        prompt, response = self._build_prompt_response(sample)

        if response:
            eos_text = self._get_eos_text()
            full_text = f"{prompt}{response}{eos_text}"
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