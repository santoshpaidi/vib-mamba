import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import WaymoMultiModalDataset
from models.vib_mamba_hf import WaymoMultiModalMamba
from models.vib_transformer import WaymoMultiModalTransformer

def compute_errors(preds, targets):
    """
    Computes errors. On nuScenes (zero-shot), targets are often dummy placeholders (zeros),
    so these metrics represent 'displacement from origin' rather than true error.
    """
    if targets.shape != preds.shape:
        targets = torch.zeros_like(preds)
        
    l2_distances = torch.norm(preds - targets, dim=-1) 
    ade = l2_distances.mean(dim=1) 
    fde = l2_distances[:, -1]      
    return ade, fde

def classify_kinematics_from_preds(preds):
    """
    Since Ground Truth is missing in zero-shot transfer, we bucket the 
    performance based on the MODEL'S INTENDED MANEUVER.
    """
    classes = []
    for traj in preds:
        # Distance to final waypoint
        total_dist = torch.norm(traj[-1]).item()
        # Max lateral deviation in the predicted trajectory
        max_lateral = torch.max(torch.abs(traj[:, 1])).item()
        
        if total_dist < 1.5: 
            classes.append("static")
        elif max_lateral > 2.0: 
            classes.append("curved")
        else: 
            classes.append("linear")
    return classes

def evaluate_model(model, dataloader, device):
    model.eval()
    results = {k: {"ade": [], "fde": []} for k in ["static", "linear", "curved", "overall"]}
    
    with torch.no_grad():
        for x_cam, x_ego, y_traj in tqdm(dataloader, desc="Evaluating", leave=False):
            x_cam, x_ego, y_traj = x_cam.to(device), x_ego.to(device), y_traj.to(device)
            
            # Forward pass
            preds, _ = model(x_cam, x_ego)
            
            # 🛠️ Fix for models returning flat outputs or different sequence lengths
            if preds.dim() != 3 or preds.shape[1] != 20:
                preds = preds.view(-1, 20, 2)
            
            ade, fde = compute_errors(preds, y_traj)
            
            # 🎯 Key Patch: Classify based on PREDS because targets are dummy zeros
            k_classes = classify_kinematics_from_preds(preds)
            
            for i, k_class in enumerate(k_classes):
                results[k_class]["ade"].append(ade[i].item())
                results[k_class]["fde"].append(fde[i].item())
                results["overall"]["ade"].append(ade[i].item())
                results["overall"]["fde"].append(fde[i].item())
                
    summary = {}
    for k in results.keys():
        if len(results[k]["ade"]) > 0:
            summary[k] = {
                "count": len(results[k]["ade"]), 
                "ade": np.mean(results[k]["ade"]), 
                "fde": np.mean(results[k]["fde"])
            }
        else:
            summary[k] = {"count": 0, "ade": 0.0, "fde": 0.0}
    return summary

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌍 Initializing Cross-Dataset Evaluation (nuScenes) on {device}...")
    
    # nuScenes processed shards should be pointed to via --data_dir
    test_dataset = WaymoMultiModalDataset(args.data_dir, split="val", history_len=10, max_shards=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Define which weights to test from the checkpoints directory
    experiments = [
        {"name": "MAMBA_LATE_FUSION", "type": "mamba", "mode": "late_fusion", "path": "best_model_late_fusion.pth"},
        {"name": "TRANSFORMER_FULL", "type": "transformer", "mode": "full", "path": "best_transformer_full.pth"},
        {"name": "MAMBA_FULL", "type": "mamba", "mode": "full", "path": "best_model_full.pth"}
    ]
    
    final_tables = {}
    
    for exp in experiments:
        # Search in top-level or ablation subfolders
        weights_path = os.path.join(args.checkpoints_dir, exp["path"])
        if not os.path.exists(weights_path):
            # Try subfolder matching the ablation mode
            weights_path = os.path.join(args.checkpoints_dir, exp["mode"], exp["path"])
            
        if not os.path.exists(weights_path):
            print(f"⚠️ Skipping {exp['name']}: Weights not found at {weights_path}")
            continue
            
        print(f"\n🧪 Testing Model: {exp['name']}")
        if exp["type"] == "mamba":
            model = WaymoMultiModalMamba(ablation_mode=exp["mode"], d_model=128, z_dim=128).to(device)
        else:
            model = WaymoMultiModalTransformer(ablation_mode=exp["mode"], d_model=128, z_dim=128).to(device)
            
        state_dict = torch.load(weights_path, map_location=device)
        # Handle DataParallel prefix and strict loading for cross-dataset flexibility
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
        
        final_tables[exp["name"]] = evaluate_model(model, test_loader, device)
        
    print("\n" + "="*85)
    print("🏁 NU-SCENES ZERO-SHOT BEHAVIORAL ANALYSIS")
    print("="*85)
    print(f"{'Model Architecture':<22} | {'Maneuver':<10} | {'Avg Dist (m)':<12} | {'Final Dist (m)':<12}")
    print("-" * 85)
    
    for name, summary in final_tables.items():
        for bucket in ["overall", "static", "linear", "curved"]:
            row = summary[bucket]
            prefix = name if bucket == "overall" else ""
            print(f"{prefix:<22} | {bucket:<10} | {row['ade']:<12.4f} | {row['fde']:<12.4f}")
        print("-" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to processed nuScenes .npz shards")
    parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints", help="Path to Waymo-trained weights")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    main(parser.parse_args())
