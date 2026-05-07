import os
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import gc

class WaymoMultiModalDataset(Dataset):
    def __init__(self, data_dir, split="train", history_len=10, max_shards=None):
        self.history_len = history_len
        self.samples = []
        
        self.all_embeddings = {}
        self.all_egos = {}
        self.all_trajectories = {}
        
        # Recursive search to bypass Kaggle's nested folders
        search_path = os.path.join(data_dir, split, "**", "*.npz")
        all_files = glob.glob(search_path, recursive=True)
        
        # Seeded Random Selection for Validation Diversity
        if max_shards is not None and len(all_files) > max_shards:
            random.seed(42) # Guarantees all ablation models use the exact same validation shards
            all_files = random.sample(all_files, max_shards)
            
        print(f"🚀 Found {len(all_files)} {split.upper()} shards in {os.path.join(data_dir, split)}")
        
        for filepath in tqdm(all_files, desc=f"Pre-loading {split.upper()} RAM"):
            shard_name = os.path.basename(filepath)
            
            with np.load(filepath) as data:
                # Do NOT cast embeddings to float() here to save ~20GB of RAM
                embs = torch.from_numpy(data['embeddings'])     
                egos = torch.from_numpy(data['ego_states']).float()     
                trajs = torch.from_numpy(data['trajectories']).float()  
                
                if embs.shape[0] < self.history_len:
                    continue
                    
                self.all_embeddings[shard_name] = embs
                self.all_egos[shard_name] = egos
                self.all_trajectories[shard_name] = trajs
                
                num_frames = embs.shape[0]
                for start_f in range(num_frames - self.history_len + 1):
                    self.samples.append((shard_name, start_f))
                    
        gc.collect() # Force memory cleanup
        print(f"✅ Pre-loading complete for {split}. Total windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard, start = self.samples[idx]
        
        # Cast embeddings to float32 only for the current batch
        x_cam = self.all_embeddings[shard][start : start + self.history_len].float()
        x_ego = self.all_egos[shard][start : start + self.history_len]
        y_traj = self.all_trajectories[shard][start + self.history_len - 1]
        
        return x_cam, x_ego, y_traj