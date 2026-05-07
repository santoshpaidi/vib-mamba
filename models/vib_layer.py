import torch
import torch.nn as nn

class VIBLayer(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, z_dim)
        self.fc_logvar = nn.Linear(input_dim, z_dim)

    def forward(self, x):
        # 1. Calculate Mean and Log-Variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # 2. Reparameterization Trick (allows backprop through randomness)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 3. Calculate KL Divergence Penalty internally
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        return z, kl_loss