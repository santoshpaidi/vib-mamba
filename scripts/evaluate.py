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
    l2_distances = torch.norm(preds - targets, dim=-1) 
    ade = l2_distances.mean(dim=1) 
    fde = l2_distances[:, -1]      
    return ade, fde

def classify_kinematics(targets):
    classes = []
    for traj in targets:
        total_dist = torch.norm(traj[-1]).item()
        max_lateral = torch.max(torch.abs(traj[:, 1])).item()
        
        if total_dist < 1.5: classes.append("static")
        elif max_lateral > 2.0: classes.append("curved")
        else: classes.append("linear")
    return classes

def evaluate_model(model, dataloader, device):
    model.eval()
    results = {k: {"ade": [], "fde": []} for k in ["static", "linear", "curved", "overall"]}
    
    with torch.no_grad():
        for x_cam, x_ego, y_traj in tqdm(dataloader, desc="Evaluating", leave=False):
            x_cam, x_ego, y_traj = x_cam.to(device), x_ego.to(device), y_traj.to(device)
            preds, _ = model(x_cam, x_ego)
            ade, fde = compute_errors(preds, y_traj)
            k_classes = classify_kinematics(y_traj)
            
            for i, k_class in enumerate(k_classes):
                results[k_class]["ade"].append(ade[i].item())
                results[k_class]["fde"].append(fde[i].item())
                results["overall"]["ade"].append(ade[i].item())
                results["overall"]["fde"].append(fde[i].item())
                
    summary = {}
    for k in results.keys():
        if len(results[k]["ade"]) > 0:
            summary[k] = {"count": len(results[k]["ade"]), "ade": np.mean(results[k]["ade"]), "fde": np.mean(results[k]["fde"])}
        else:
            summary[k] = {"count": 0, "ade": 0.0, "fde": 0.0}
    return summary

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Master Evaluation Pipeline on {device}...")
    
    test_dataset = WaymoMultiModalDataset(args.data_dir, split="val", history_len=10, max_shards=None)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    experiments = [
        {"name": "MAMBA_FRONT", "type": "mamba", "mode": "front_only", "path": "best_model_front_only.pth"},
        {"name": "MAMBA_8_CAM", "type": "mamba", "mode": "8_cam", "path": "best_model_8_cam.pth"},
        {"name": "MAMBA_LATE_FUSION", "type": "mamba", "mode": "late_fusion", "path": "best_model_late_fusion.pth"},
        {"name": "TRANSFORMER_FULL", "type": "transformer", "mode": "full", "path": "best_transformer_full.pth"},
        {"name": "MAMBA_NO_VIB", "type": "mamba", "mode": "full", "path": "mamba_no_vib/full/best_model_full.pth"}, # <--- ADD THIS
        {"name": "MAMBA_FULL (OURS)", "type": "mamba", "mode": "full", "path": "best_model_full.pth"}
    ]
    
    final_tables = {}
    
    for exp in experiments:
        weights_path = os.path.join(args.checkpoints_dir, exp["path"])
        if not os.path.exists(weights_path):
            continue
            
        print(f"\n{'='*50}\n🧪 EVALUATING: {exp['name']}\n{'='*50}")
        if exp["type"] == "mamba":
            model = WaymoMultiModalMamba(ablation_mode=exp["mode"], d_model=128, z_dim=128).to(device)
        else:
            model = WaymoMultiModalTransformer(ablation_mode=exp["mode"], d_model=128, z_dim=128).to(device)
            
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=True)
        
        final_tables[exp["name"]] = evaluate_model(model, test_loader, device)
        
    print("\n\n🏆 MASTER KINEMATIC STRATIFICATION RESULTS 🏆")
    print("="*85)
    print(f"{'Model Architecture':<22} | {'Kinematic':<10} | {'ADE (m)':<10} | {'FDE (m)':<10}")
    print("-" * 85)
    
    for exp in experiments:
        name = exp["name"]
        if name not in final_tables: continue
        summary = final_tables[name]
        print(f"{name:<22} | {'Overall':<10} | {summary['overall']['ade']:<10.4f} | {summary['overall']['fde']:<10.4f}")
        print(f"{'':<22} | {'Static':<10} | {summary['static']['ade']:<10.4f} | {summary['static']['fde']:<10.4f}")
        print(f"{'':<22} | {'Linear':<10} | {summary['linear']['ade']:<10.4f} | {summary['linear']['fde']:<10.4f}")
        print(f"{'':<22} | {'Curved':<10} | {summary['curved']['ade']:<10.4f} | {summary['curved']['fde']:<10.4f}")
        print("-" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints")
    main(parser.parse_args())