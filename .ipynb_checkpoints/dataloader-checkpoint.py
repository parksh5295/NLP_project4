from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64, is_train=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def get_train_loader(tokenizer, batch_size):
    train_file = './dataset/train.csv'
    train_data = pd.read_csv(train_file)
    assert 'text' in train_data.columns and 'label' in train_data.columns, '[Train] CSV file error...'

    valid_file = './dataset/valid.csv'
    valid_data = pd.read_csv(valid_file)
    assert 'text' in train_data.columns and 'label' in train_data.columns, '[Valid] CSV file error...'


    train_text = train_data['text'].tolist()
    valid_text = valid_data['text'].tolist()

    train_label = train_data['label'].tolist()
    valid_label = valid_data['label'].tolist()

    train_dataset = TextDataset(train_text, train_label, tokenizer)
    valid_dataset = TextDataset(valid_text, valid_label, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = 16, shuffle=False)
    
    
    return train_loader, valid_loader

def get_test_loader(tokenizer,is_test=False):
    test_file = './dataset/test.csv' if is_test else './dataset/valid.csv'
    test_data = pd.read_csv(test_file)
    assert 'text' in test_data.columns and 'label' in test_data.columns, '[Test] CSV file error...'

    test_text = test_data['text'].tolist()
    test_label = test_data['label'].tolist()

    test_dataset = TextDataset(test_text, test_label, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)
    
    return test_loader