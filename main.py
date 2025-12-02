import argparse
import os

from architecture import *
from dataloader import *
from train import *
from test import *


def main(args):
    model, tokenizer = get_model()
    train_loader, valid_loader = get_train_loader(tokenizer, args.batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    train(model, optimizer, loss_fn, args.epochs, train_loader, valid_loader)


if __name__ == '__main__':
    os.makedirs('./results/', exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    main(args)
    
    
