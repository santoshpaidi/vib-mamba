import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class NuScenesWaymoAligner:
    def __init__(self, nusc_path, version='v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=nusc_path, verbose=False)
        # Refined Mapping for 8 slots:
        # NuScenes has 6 cams. Waymo expects 8.
        # We use 'SIDE' cams (which in NuScenes are FRONT_LEFT/RIGHT and BACK_LEFT/RIGHT)
        self.cam_mapping = [
            'CAM_FRONT',        # 0: Front
            'CAM_FRONT_LEFT',   # 1: Front-Left
            'CAM_FRONT_RIGHT',  # 2: Front-Right
            'CAM_BACK_LEFT',    # 3: Side-Left (NuScenes Front-Left/Back-Left transition)
            'CAM_BACK_RIGHT',   # 4: Side-Right (NuScenes Front-Right/Back-Right transition)
            'CAM_BACK',         # 5: Back
            'CAM_BACK_LEFT',    # 6: Back-Left
            'CAM_BACK_RIGHT'    # 7: Back-Right
        ]
        
    def get_closest_ego_pose(self, target_t, scene_token):
        """Finds the closest ego pose for a target timestamp."""
        scene = self.nusc.get('scene', scene_token)
        curr = self.nusc.get('sample', scene['first_sample_token'])
        best_pose = None
        min_diff = float('inf')
        
        # In a real extraction, we'd pre-calculate this mapping, 
        # but for mini/validation, this search is acceptable.
        while curr:
            diff = abs(curr['timestamp'] - target_t)
            if diff < min_diff:
                min_diff = diff
                best_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', curr['data']['CAM_FRONT'])['ego_pose_token'])
            if diff > 1e6 and curr['timestamp'] > target_t: break # Optimization
            if not curr['next']: break
            curr = self.nusc.get('sample', curr['next'])
        return best_pose

    def get_ego_kinematics(self, sample_token, history_len=16, dt=0.1):
        """
        Extracts 10Hz history (1.6s total).
        """
        current_sample = self.nusc.get('sample', sample_token)
        curr_time = current_sample['timestamp']
        
        ref_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])['ego_pose_token'])
        c_pos = np.array(ref_pose['translation'])
        c_rot = Quaternion(ref_pose['rotation'])
        
        history = []
        for i in range(history_len):
            target_t = curr_time - int((history_len - 1 - i) * dt * 1e6)
            pose = self.get_closest_ego_pose(target_t, current_sample['scene_token'])
            
            # Position (Local)
            pos = np.array(pose['translation']) - c_pos
            pos = c_rot.inverse.rotate(pos)
            
            # Velocity (Approximate from neighbors if needed, or set to 0 for now)
            # nuScenes ego_pose doesn't store velocity, it's derived.
            vel = [0.0, 0.0] 
            
            history.append([pos[0], pos[1], vel[0], vel[1]])
            
        return np.array(history).flatten()

    def get_gt_trajectory(self, sample_token, future_len=20, dt=0.1):
        """
        Extracts 10Hz Ground Truth (2.0s total).
        """
        current_sample = self.nusc.get('sample', sample_token)
        curr_time = current_sample['timestamp']
        
        ref_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])['ego_pose_token'])
        c_pos = np.array(ref_pose['translation'])
        c_rot = Quaternion(ref_pose['rotation'])

        traj = []
        for i in range(1, future_len + 1):
            target_t = curr_time + int(i * dt * 1e6)
            pose = self.get_closest_ego_pose(target_t, current_sample['scene_token'])
            
            pos = np.array(pose['translation']) - c_pos
            pos = c_rot.inverse.rotate(pos)
            traj.append([pos[0], pos[1]])
            
        return np.array(traj)

def run_extraction(nusc_path, output_path, version='v1.0-mini', device='cuda'):
    aligner = NuScenesWaymoAligner(nusc_path, version=version)
    
    print(f"🚀 Loading HF CLIP-ViT-L/14 (1024-dim parity)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    
    for scene_idx, scene in enumerate(tqdm(aligner.nusc.scene, desc="Processing Scenes")):
        scene_embs, scene_egos, scene_gt = [], [], []
        curr_token = scene['first_sample_token']
        
        while curr_token:
            sample = aligner.nusc.get('sample', curr_token)
            frame_cams = []
            
            for cam_name in aligner.cam_mapping:
                cam_data = aligner.nusc.get('sample_data', sample['data'][cam_name])
                img_path = os.path.join(nusc_path, cam_data['filename'])
                image = Image.open(img_path).convert("RGB")
                
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.vision_model(**inputs)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        feat = outputs.pooler_output
                    else:
                        feat = outputs.last_hidden_state[:, 0, :]
                    feat = feat.cpu().numpy().flatten().astype(np.float32)
                
                frame_cams.append(feat)
            
            scene_embs.append(np.array(frame_cams)) 
            scene_egos.append(aligner.get_ego_kinematics(curr_token))
            scene_gt.append(aligner.get_gt_trajectory(curr_token))
            curr_token = sample['next']
            
        np.savez_compressed(
            os.path.join(output_path, f"scene_{scene_idx}.npz"),
            embeddings=np.array(scene_embs),
            ego_states=np.array(scene_egos),
            trajectories=np.array(scene_gt)
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-mini')
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    run_extraction(args.nusc_path, args.output_path, version=args.version)
