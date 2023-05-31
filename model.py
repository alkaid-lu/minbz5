import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, feature):
        latent = self.encoder(feature)
        reconstruct = self.decoder(latent)
        return reconstruct, latent


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            
        )

    def forward(self, feature):
        latent = self.encoder(feature)
        mean, var = torch.chunk(latent, 2, dim=-1)

        std = var.mul(0.5).exp_()
        esp = torch.randn(*mean.size())
        latent_sample = mean + std * esp

        reconstruct= self.decoder(latent_sample)
        return reconstruct, latent

    def sample(self, num_samples):
        with torch.no_grad():
            latent_sample = torch.randn(num_samples, self.latent_dim)
            reconstruct = self.decoder(latent_sample)
        return reconstruct


def make_model(model_name):
    if model_name == 'AE':
        return AutoEncoder(12, 32, 3)
    if model_name == 'VAE':
        return VariationalAutoEncoder(12, 32, 3)