import torch
from torch import optim
from tqdm import tqdm
from eval import test


def train(model, optimizer, loss_fn, epochs, train_loader, valid_loader):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        
        accuracy = test(model, valid_loader)
        print(f"Epoch {epoch+1} Train Loss {total_loss / len(train_loader)}")
        print(f"Validation performance {accuracy}")

        if accuracy >= best_acc:
            best_acc = accuracy
            print("Best model saved...")
            torch.save(model.state_dict(), './results/best_model.pt')

    torch.save(model.state_dict(), './results/end_model.pt')

    return model