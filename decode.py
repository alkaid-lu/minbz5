# coding=utf-8
import torch
from data import Monecular
from model import make_model
from torch.utils.data import DataLoader

#加载训练好的模型权重
model = make_model('VAE')
checkpoint = torch.load('./model.pth')
model.load_state_dict(checkpoint, strict=False)

#加载样本数据点
dataset = Monecular('sample_data.csv')
test_loader = DataLoader(dataset)
idx2char = {value: key for key, value in dataset.char2idx.items()}

model.eval()

for feature, label_word in test_loader:
    reconstructed_output_list, properties_output_list = model(feature)
    print(label_word[0])
    for reconstructed_output, properties_output in zip(reconstructed_output_list, properties_output_list):
        feature = feature.squeeze()
        reconstructed_output = reconstructed_output.squeeze()

        pred = torch.argmax(reconstructed_output, dim=1)
        pred = pred.tolist()
        pred_word = []

        for v in pred:
            pred_word.append(idx2char[v])

        # print(label_word[0], '--->', ''.join(pred_word))
        print('--->', ''.join(pred_word))

