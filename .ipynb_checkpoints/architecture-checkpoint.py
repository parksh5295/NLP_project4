from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn

class CustomClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

def get_model():
    
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    model = CustomClassifier(base_model, 2).to('cuda')
    
    return model, tokenizer
    
