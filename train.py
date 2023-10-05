import argparse
import torch
from torch.utils.data import DataLoader
from data import Monecular
from model import make_model


def train(model, data_loader, optimzizer, epochs, scheduler):
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        loss_val = 0
        for i, (feature, properties) in enumerate(train_loader):
            pred_smile, pred_prop  = model(feature)

            reconstruct_loss = criterion(feature, pred_smile)
            ce_loss = criterion(properties, pred_prop)

            loss = reconstruct_loss + ce_loss * 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        print(f'training loss for epoch {epoch} is {loss_val}')
        scheduler.step()
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'model.pth.tar')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['VAE', 'AE'], default='VAE')
    parser.add_argument('--smile_length', type=int, default=0)
    args = parser.parse_args()

    dataset = Monecular('sample_data.csv', max_length=args.smile_length)
    train_loader = DataLoader(dataset)

    model = make_model(args.model_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    epochs = 500

    train(model, train_loader, optimizer, epochs, scheduler)