import torch
import argparse
from architecture import *
from dataloader import *
from sklearn.metrics import accuracy_score


def test(model, test_loader):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
        
    accuracy = accuracy_score(test_labels, test_preds)
    return accuracy
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--is_test', action='store_true')
    args = parser.parse_args()
    
    
    model, tokenizer = get_model()
    test_loader = get_test_loader(tokenizer, is_test=args.is_test)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to('cuda')
    
    
    accuracy = test(model, test_loader)
    print(f"{accuracy}")