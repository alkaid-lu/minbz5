# coding=utf-8
import torch
import pandas
from data import Monecular
from model_distance import make_model
from torch.utils.data import DataLoader
#from train1 import test_dataset

model = make_model('VAE')
checkpoint = torch.load('model1226.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])

test_dataset = Monecular('1227verify_data.csv', vocab='vocab.list')
inverse_dict = test_dataset.getinverse()
test_loader = DataLoader(test_dataset)
idx2char = {value: key for key, value in test_dataset.char2idx.items()}

model.eval()
for feature, label_word in test_loader:
    #feature = torch.tensor(feature)
    prediction, _ = model(feature)    
    feature = feature.squeeze()
    #prediction = torch.tensor(prediction)
    pred_list = []
    for pred in prediction:
        pred = torch.argmax(pred, dim=-1)
        pred = pred.tolist()[0]
        pred_word = []

        for v in pred:
            pred_word.append(idx2char[v])
        pred_list.append(''.join(pred_word))
    print(label_word[0], '--->\n', '\n'.join(pred_list))
    print('==============================')
