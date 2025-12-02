from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn

class CustomClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids = input_ids, attention_mask = attention_mask)
        
        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return logits

def get_model():
    
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    model = CustomClassifier(base_model, 2).to('cuda')
    
    return model, tokenizer
    
