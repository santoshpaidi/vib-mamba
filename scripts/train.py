import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import WaymoMultiModalDataset
from models.vib_mamba_hf import WaymoMultiModalMamba

def compute_loss(pred_traj, target_traj, kl_loss, beta=1e-4):
    """
    Computes the total loss = MSE(Reconstruction) + Beta * KL_Divergence
    """
    mse = nn.MSELoss()(pred_traj, target_traj)
    # kl_loss is returned per-batch from the VIB layer, so we take the mean
    total_kl = kl_loss.mean()
    total_loss = mse + (beta * total_kl)
    
    return total_loss, mse, total_kl

def train_epoch(model, dataloader, optimizer, device, beta):
    model.train()
    total_loss, total_mse, total_kl = 0, 0, 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for x_cam, x_ego, y_traj in pbar:
        x_cam, x_ego, y_traj = x_cam.to(device), x_ego.to(device), y_traj.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (handles both Mamba and Transformer identically)
        preds, kl_loss = model(x_cam, x_ego)
        
        # Calculate losses
        loss, mse, kl = compute_loss(preds, y_traj, kl_loss, beta)
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping to prevent explosion in early epochs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_kl += kl.item()
        
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'MSE': f"{mse.item():.4f}"})
        
    return total_loss / len(dataloader), total_mse / len(dataloader)

def validate_epoch(model, dataloader, device, beta):
    model.eval()
    total_loss, total_mse = 0, 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for x_cam, x_ego, y_traj in pbar:
            x_cam, x_ego, y_traj = x_cam.to(device), x_ego.to(device), y_traj.to(device)
            
            preds, kl_loss = model(x_cam, x_ego)
            loss, mse, _ = compute_loss(preds, y_traj, kl_loss, beta)
            
            total_loss += loss.item()
            total_mse += mse.item()
            
    return total_loss / len(dataloader), total_mse / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training Pipeline on {device}...")
    
    # --- 1. SETUP DIRECTORIES ---
    save_path = os.path.join(args.save_dir, args.ablation_mode)
    os.makedirs(save_path, exist_ok=True)
    
    # --- 2. DATA LOADING ---
    print(f"\n📦 Loading Data from {args.data_dir}...")
    # Training set
    train_dataset = WaymoMultiModalDataset(args.data_dir, split="train", history_len=args.history_len, max_shards=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    # Validation set (Capped at 30 shards for RAM safety during evaluation loop)
    val_dataset = WaymoMultiModalDataset(args.data_dir, split="val", history_len=args.history_len, max_shards=30)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- 3. MODEL INITIALIZATION ---
    print(f"\n🧠 Initializing {args.model_type.upper()} Architecture (Mode: {args.ablation_mode})...")
    
    if args.model_type == "mamba":
        model = WaymoMultiModalMamba(
            ablation_mode=args.ablation_mode, 
            d_model=args.d_model, 
            z_dim=args.z_dim, 
            future_steps=args.future_steps
        ).to(device)
    elif args.model_type == "transformer":
        from models.vib_transformer import WaymoMultiModalTransformer
        model = WaymoMultiModalTransformer(
            ablation_mode=args.ablation_mode, 
            d_model=args.d_model, 
            z_dim=args.z_dim, 
            future_steps=args.future_steps
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Multi-GPU Support
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)

    # --- 4. OPTIMIZATION ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # --- 5. TRAINING LOOP ---
    best_val_mse = float('inf')
    
    print("\n" + "="*60)
    print(f"🏁 STARTING TRAINING: {args.model_type.upper()} | {args.ablation_mode.upper()}")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_mse = train_epoch(model, train_loader, optimizer, device, args.beta)
        val_loss, val_mse = validate_epoch(model, val_loader, device, args.beta)
        
        scheduler.step(val_mse)
        
        print(f"📊 [Epoch {epoch}] Train Loss: {train_loss:.4f} | Train MSE: {train_mse:.4f}")
        print(f"📈 [Epoch {epoch}] Val Loss:   {val_loss:.4f} | Val MSE:   {val_mse:.4f}")
        
        # Save checkpoint if it's the best so far
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            
            # Differentiate file names so Mamba and Transformer don't overwrite each other
            prefix = "best_transformer" if args.model_type == "transformer" else "best_model"
            model_name = f"{prefix}_{args.ablation_mode}.pth"
            
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            print(f"💾 New Best Model Saved! (Val MSE: {best_val_mse:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data & Paths
    parser.add_argument('--data_dir', type=str, required=True, help="Path to pre-extracted embeddings")
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help="Where to save model weights")
    
    # Architecture
    parser.add_argument('--model_type', type=str, default="mamba", choices=['mamba', 'transformer'], help="Which temporal backbone to use")
    parser.add_argument('--ablation_mode', type=str, default="full", choices=['front_only', '8_cam', 'full', 'late_fusion'], help="Which data fusion mode to use")
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1e-4, help="VIB KL-Divergence Loss Weight")
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model Dimensions
    parser.add_argument('--history_len', type=int, default=10, help="Number of past timesteps")
    parser.add_argument('--future_steps', type=int, default=20, help="Number of future timesteps to predict")
    parser.add_argument('--d_model', type=int, default=128, help="Hidden dimension size")
    parser.add_argument('--z_dim', type=int, default=128, help="VIB Latent dimension size")
    
    main(parser.parse_args())