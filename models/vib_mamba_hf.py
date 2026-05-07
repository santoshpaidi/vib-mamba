import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel
from .vib_layer import VIBLayer

class WaymoMultiModalMamba(nn.Module):
    def __init__(self, ablation_mode="full", d_model=128, z_dim=128, future_steps=20):
        super().__init__()
        self.mode = ablation_mode
        self.future_steps = future_steps
        
        if self.mode == "front_only":
            self.vision_proj = nn.Linear(1024, d_model)
        else:
            self.vision_proj = nn.Sequential(
                nn.Linear(8 * 1024, 512),
                nn.GELU(),
                nn.Linear(512, d_model)
            )
            
        if self.mode in ["full", "late_fusion"]:
            self.ego_proj = nn.Linear(64, d_model)
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU()
            )
            
        config = MambaConfig(hidden_size=d_model, num_hidden_layers=4)
        self.mamba = MambaModel(config)
        self.vib = VIBLayer(d_model, z_dim)
        
        self.proj_out = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * 2)
        )

    def forward(self, x_cam, x_ego):
        batch_size, seq_len = x_cam.shape[0], x_cam.shape[1]
        
        if self.mode == "front_only":
            v_feat = x_cam[:, :, 0, :]
            v_feat = self.vision_proj(v_feat)
            fused_feat = v_feat
            
        elif self.mode == "8_cam":
            v_feat = x_cam.view(batch_size, seq_len, -1)
            v_feat = self.vision_proj(v_feat)
            fused_feat = v_feat
            
        elif self.mode == "full":
            v_feat = x_cam.view(batch_size, seq_len, -1)
            v_feat = self.vision_proj(v_feat)
            e_feat = self.ego_proj(x_ego)
            
            concat_feat = torch.cat([v_feat, e_feat], dim=-1)
            fused_feat = self.fusion_layer(concat_feat)
            
        elif self.mode == "late_fusion":
            v_feat = x_cam.view(batch_size, seq_len, -1)
            v_feat = self.vision_proj(v_feat)
            
            # Mamba processes vision BEFORE fusion
            mamba_out = self.mamba(inputs_embeds=v_feat).last_hidden_state
            last_vis_hidden = mamba_out[:, -1, :] 
            
            # Extract final timestep of kinematics
            e_feat = self.ego_proj(x_ego)
            last_ego = e_feat[:, -1, :]
            
            # Fuse explicitly at the bottleneck
            concat_feat = torch.cat([last_vis_hidden, last_ego], dim=-1)
            last_hidden = self.fusion_layer(concat_feat)
            
            # Bypass the standard Mamba output extraction below
            z, kl_loss = self.vib(last_hidden)
            out = self.proj_out(z).view(-1, self.future_steps, 2)
            return out, kl_loss
            
        # Standard Mamba processing for front_only, 8_cam, and full
        mamba_out = self.mamba(inputs_embeds=fused_feat).last_hidden_state
        last_hidden = mamba_out[:, -1, :]
        
        z, kl_loss = self.vib(last_hidden)
        
        out = self.proj_out(z)
        out = out.view(-1, self.future_steps, 2)
        
        return out, kl_loss