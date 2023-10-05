import pandas
import torch
from torch.nn import ConstantPad1d
from torch.utils.data import Dataset
import torch.nn.functional as F

SMILE = 3
DENSITY = 5
CALORICITY = 6
MELTING = 7


class Monecular(Dataset):
    def __init__(self, datapath, max_length=0):
        self.samples = pandas.read_csv(datapath, header=0)
        self.vocab = set()
        self.max_length = max_length

        smile_words = self.samples['SMILES']
        for w in smile_words:
            for c in w:
                self.vocab.add(c)
            
            if len(w) > self.max_length:
                self.max_length = len(w)
        
        self.vocab = list(self.vocab)
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        char_list = [self.char2idx[char] for char in sample["SMILES"]]

        feature = torch.tensor(char_list)
        feature = F.one_hot(feature, num_classes=len(self.vocab)).to(torch.float32)
        padding = (0, 0, 0, self.max_length - len(char_list))
        feature = F.pad(feature, padding, value=-1)

        density = (sample["DENSITY"] - 0.75) / 0.75
        caloricity = (sample["CALORICITY"] - 25) / 25
        melting = (sample["MELTING"] - 75) / 225

        properties = torch.tensor([density, caloricity, melting]).float()
        
        return feature, properties

