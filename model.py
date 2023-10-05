import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_property):
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

        self.property_predictor = nn.Linear(
            latent_dim, 
            num_property
        )


    def forward(self, feature):
        latent = self.encoder(feature)
        reconstruct = self.decoder(latent)
        properties = self.property_predictor(latent)
        return reconstruct, properties


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_property):
        super(VariationalAutoEncoder, self).__init__()

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
            nn.Sigmoid()
        )

        self.property_predictor = nn.Linear(
            latent_dim, 
            num_property
        )

    def __decode_latent(self, latent):
        mean, var = torch.chunk(latent, 2, dim=-1)
        std = var.mul(0.5).exp_()
        esp = torch.randn(*mean.size())
        latent_sample = mean + std * esp

        reconstruct= self.decoder(latent_sample)
        properties = self.property_predictor(latent_sample)

        return reconstruct, properties

    def forward(self, feature):
        latent = self.encoder(feature)
        reconstruct, properties = self.__decode_latent(latent)
        if self.training:
            return self.__decode_latent(latent)

        else:
            reconstruct_list, properties_list = [], []
            for i in range(5):
                reconstruct, properties = self.__decode_latent(latent)

                reconstruct_list.append(reconstruct)
                properties_list.append(properties)
            return reconstruct_list, properties_list

    def decode_with_fixed_distance(self, mean, distance):
        # 规范化潜在表示
        normalized_mean = mean / torch.norm(mean, dim=1, keepdim=True)
        # 计算新的点
        distance = 1.0
        new_mean = normalized_mean * distance
        # 解码新的点
        reconstructed_output = self.decoder(new_mean)
        properties_output = self.property_predictor(new_mean)
        return reconstructed_output, properties_output

    def forward_with_fixed_distance(self, feature, distance):
        #运行模型并获取重构输出
        reconstructed_output, properties_output = self.forward(feature)
        #在隐空间内设置固定距离的解码点
        distance = 1.0
        decoded_with_fixed_distance = self.decode_with_fixed_distance(sample_mean, distance)
        if self.training:
            return self.__decode_latent(latent)

        else:
            reconstructed_output_list, properties_output_list = [], []
            for i in range(5):
                reconstructed_output, properties_output = self.decode_with_fixed_distance(decoded_with_fixed_distance)

                reconstructed_output_list.append(reconstructed_output)
                properties_output_list.append(properties_output)
            return reconstructed_output_list, properties_output_list

        
def make_model(model_name):
    if model_name == 'AE':
        return AutoEncoder(12, 32, 3, 3)
    if model_name == 'VAE':
        return VariationalAutoEncoder(12, 32, 3, 3)