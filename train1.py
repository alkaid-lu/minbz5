import argparse
import torch
from torch.utils.data import DataLoader
from data import Monecular
from model import make_model
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import pickle
import pandas as pd


def train(model, data_loader, optimzizer, epochs, scheduler):
    criterion = torch.nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        loss_val = 0
        
        for i, (feature) in enumerate(train_loader):
            if i % 100 == 0:
                print(f"entering {i}th batch...")
            feature = feature.cuda()
            pred_smile, pred_prop  = model(feature)
            mask = (feature != -1).float().cuda()
            masked_feature = feature * mask
            masked_predict = pred_smile * mask
            #import pdb
            #pdb.set_trace()
            reconstruct_loss = criterion(masked_feature, masked_predict)
            # ce_loss = criterion(properties, pred_prop)

            loss = reconstruct_loss #+ ce_loss * 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        losses.append(loss_val)
        print(f'training loss for epoch {epoch} is {loss_val}')
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.show()
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'model1226.pth.tar')

def test(model, data_loader, optimizer, epochs, scheduler):
    criterion = torch.nn.MSELoss()
    test_losses = []
    
    for epoch in range(epochs):
        loss_val = 0
        for i, (feature) in enumerate(data_loader):
            if i % 100 == 0:
                print(f"entering {i}th batch...")
            feature = feature.cuda()
            pred_smile, pred_prop = model(feature)
            mask = (feature != -1).float().cuda()
            masked_feature = feature * mask
            masked_predict = pred_smile * mask
            reconstruct_loss = criterion(masked_feature, masked_predict)
            # ce_loss = criterion(properties, pred_prop)
            loss = reconstruct_loss  # + ce_loss * 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        test_losses.append(loss_val)
        print(f'test loss for epoch {epoch} is {loss_val}')
    plt.plot(test_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Epoch')
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['VAE', 'AE'], default='VAE')
    parser.add_argument('--smile_length', type=int, default=0)
    args = parser.parse_args()
    
    dataset = Monecular('all_VAE_data.csv', max_length=args.smile_length)
    train_ratio = 0.95
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset,batch_size=5000, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=2000, num_workers=1)  
     
    model = make_model(args.model_type)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    epochs = 35

    train(model, train_loader, optimizer, epochs, scheduler)
    test(model, test_loader, optimizer, epochs, scheduler)
