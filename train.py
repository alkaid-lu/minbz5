import argparse
import torch 
from torch.utils.data import DataLoader
from data import Monecular
from model import make_model


def train(model, data_loader, optimizer, epochs):
    criterion = torch.nn.MSELoss()
    mole = Monecular('145sample_data.csv')
    smile_dict = mole.char2idx
    for epoch in range(epochs):
        loss_val = 0
        for i, (feature, label_word) in enumerate(data_loader):
            prediction, _ = model(feature)
            loss = criterion(feature, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            prediction_1 = prediction.squeeze()
            a = torch.argmax(prediction_1, dim=1)
            a = a.tolist()
            re_smiles = []
            if epoch % 10 == 0 and i == 0:
                for v in a:
                    for key, value in smile_dict.items():
                        if value == v:
                            re_smiles.append(key)
                print(re_smiles)
        if epoch % 10 == 0:
            print(f'training loss for epoch {epoch} is {loss_val}')
        #print(prediction.size())
    # checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    # torch.save(checkpoint, 'model.pth.tar')
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['VAE', 'AE'], default='VAE')
    parser.add_argument('--smile_length', type=int, default=0)
    args = parser.parse_args()

    dataset = Monecular('145sample_data.csv', max_length=args.smile_length)
    train_loader = DataLoader(dataset)

    model = make_model(args.model_type)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 110

    train(model, train_loader, optimizer, epochs)

    # model.eval()
    # num_samples = 10
    # with torch.no_grad():
    #     reconstruct = model.sample(num_samples)
    #     print(reconstruct)
    #     print(reconstruct.size())
