import torch
import torch.nn as nn
from .vib_layer import VIBLayer

class WaymoMultiModalTransformer(nn.Module):
    def __init__(self, ablation_mode="full", d_model=128, z_dim=128, future_steps=20):
        super().__init__()
        self.mode = ablation_mode
        self.future_steps = future_steps
        
        self.vision_proj = nn.Sequential(
            nn.Linear(8 * 1024, 512),
            nn.GELU(),
            nn.Linear(512, d_model)
        )
        self.ego_proj = nn.Linear(64, d_model)
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
            
        # Standard Transformer Encoder (4 Layers to match Mamba)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4, 
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.vib = VIBLayer(d_model, z_dim)
        
        self.proj_out = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * 2)
        )

    def forward(self, x_cam, x_ego):
        batch_size, seq_len = x_cam.shape[0], x_cam.shape[1]
        
        v_feat = x_cam.view(batch_size, seq_len, -1)
        v_feat = self.vision_proj(v_feat)
        e_feat = self.ego_proj(x_ego)
        
        concat_feat = torch.cat([v_feat, e_feat], dim=-1)
        fused_feat = self.fusion_layer(concat_feat)
        
        # Transformer processing
        transformer_out = self.transformer(fused_feat)
        last_hidden = transformer_out[:, -1, :] # Grab final timestep
        
        z, kl_loss = self.vib(last_hidden)
        out = self.proj_out(z).view(-1, self.future_steps, 2)
        
        return out, kl_loss