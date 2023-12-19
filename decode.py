# coding=utf-8
import torch
import numpy as np
from data import Monecular
from model import make_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

#加载训练好的模型权重
model = make_model('VAE')
checkpoint = torch.load('./model1214.pth')
model.load_state_dict(checkpoint, strict=False)

#加载样本数据点
dataset = Monecular('remove_N_char_data.csv')
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

latent_vars = []
num_samples = 1000
with torch.no_grad():
    for _ in range(num_samples):
        latent_dim = 3
        z = torch.randn(1, latent_dim)
        new_molecule = model.decoder(z)
        latent_vars.append(z)
        #latent_vars.append(z.numpy())
#latent_vars = torch.cat(latent_vars, dim=0)
#plt.scatter(latent_vars[:, 0], latent_vars[:, 1])
#plt.xlabel('Latent Variable 1')
#plt.ylabel('Latent Variable 2')
#plt.title('Generated Molecules Distribution')
#plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#lable = np.array(latent_vars)
#ax.scatter(lable[:, 0], lable[:, 1], lable[:, 2])
#ax.set_xlabel('Latent Variable 1')
#ax.set_ylabel('Latent Variable 2')
#ax.set_zlabel('Latent Variable 3')
#ax.set_title('Generated Molecules Distribution')
#plt.show()
