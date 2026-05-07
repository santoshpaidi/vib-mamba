import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf
import torch
from tqdm.auto import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

# Ensure Waymo protos are in the path
sys.path.append(os.path.join(os.getcwd(), 'waymo-open-dataset/src'))
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    print("⚠️ Waymo protos not found. Please run 'python data/setup_protos.py' first.")
    sys.exit(1)

def extract_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing CLIP Feature Extraction on {device}...")

    # Load the frozen Spatial Encoder (CLIP-ViT-Large)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    encoder.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    tfrecord_files = sorted(glob.glob(os.path.join(args.data_dir, '*.tfrecord')))
    print(f"🔍 Found {len(tfrecord_files)} TFRecords to process.")

    for filepath in tqdm(tfrecord_files, desc="Processing TFRecords"):
        dataset = tf.data.TFRecordDataset(filepath, compression_type='')
        
        current_scenario_id = None
        scenario_embeddings = []
        frame_names = []

        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            scenario_id = frame.context.name
            timestamp = frame.timestamp_micros

            # If we transition to a new scenario, save the previous one to disk
            if current_scenario_id != scenario_id and current_scenario_id is not None:
                out_path = os.path.join(args.output_dir, f"{current_scenario_id}.npz")
                np.savez_compressed(
                    out_path, 
                    embeddings=np.stack(scenario_embeddings),
                    frame_names=np.array(frame_names)
                )
                scenario_embeddings = []
                frame_names = []

            current_scenario_id = scenario_id

            # Extract the Front Camera Image (Index 0)
            front_camera_image = frame.images[0].image
            img_tensor = tf.io.decode_jpeg(front_camera_image)
            img_np = img_tensor.numpy()

            # Process through CLIP
            inputs = processor(images=img_np, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Extract the 1024-dimensional pooler_output
                embedding = encoder(**inputs).pooler_output.squeeze(0).cpu().numpy()

            scenario_embeddings.append(embedding)
            frame_names.append(f"{scenario_id}-{timestamp}")
            
        # Catch and save the final sequence in the file
        if current_scenario_id is not None and len(scenario_embeddings) > 0:
            out_path = os.path.join(args.output_dir, f"{current_scenario_id}.npz")
            np.savez_compressed(
                out_path, 
                embeddings=np.stack(scenario_embeddings),
                frame_names=np.array(frame_names)
            )

    print(f"✅ Feature extraction complete! All embeddings saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract spatial embeddings using CLIP")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing raw Waymo .tfrecord files")
    parser.add_argument('--output_dir', type=str, default="teacher_embeddings", help="Directory to save the .npz files")
    args = parser.parse_args()
    
    extract_features(args)