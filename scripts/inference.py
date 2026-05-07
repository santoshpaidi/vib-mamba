import os
import sys
import glob
import pickle
import gc
import shutil
import tarfile
import numpy as np
import torch
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), 'waymo-open-dataset/src'))
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2

from models import get_model
from data.dataset import ChunkedTestDataset

def stream_predictions_to_disk(batch_data, file_path):
    with open(file_path, 'ab') as f:
        pickle.dump(batch_data, f)

def load_and_clear_temp_predictions(file_path):
    all_preds = []
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            while True:
                try: all_preds.extend(pickle.load(f))
                except EOFError: break
        os.remove(file_path)
    return all_preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, device)
    
    state_dict = torch.load(args.weights, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    KAGGLE_OUTPUT_DIR = '/kaggle/working/Waymo_E2E_Submission'
    STAGING_DIR = '/kaggle/working/FinalSubmission'
    TEMP_PREDS_FILE = os.path.join(KAGGLE_OUTPUT_DIR, 'streaming_predictions.pkl')

    os.makedirs(KAGGLE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STAGING_DIR, exist_ok=True)

    all_shards = sorted(glob.glob(os.path.join(args.test_dir, "*.npz")))
    chunk_size = 20
    part_count = 0
    
    for i in range(0, len(all_shards), chunk_size):
        chunk_files = all_shards[i:i + chunk_size]
        dataset = ChunkedTestDataset(chunk_files)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for imgs, frame_names in tqdm(loader, desc=f"Part {part_count}"):
                raw_preds = model(imgs.to(device)).cpu().numpy()
                batch_preds = []
                for j in range(len(frame_names)):
                    parts = str(frame_names[j]).split('-')
                    clean_frame_name = f"{parts[0]}-{parts[1]}"
                    
                    pred_proto = wod_e2ed_submission_pb2.TrajectoryPrediction(
                        pos_x=raw_preds[j, :, 0].astype(np.float32).tolist(),
                        pos_y=raw_preds[j, :, 1].astype(np.float32).tolist()
                    )
                    batch_preds.append(wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
                        frame_name=clean_frame_name, trajectory=pred_proto))

                stream_predictions_to_disk(batch_preds, TEMP_PREDS_FILE)
                del batch_preds 

        to_save = load_and_clear_temp_predictions(TEMP_PREDS_FILE)
        
        # Save Protobuf
        shard_proto = wod_e2ed_submission_pb2.E2EDChallengeSubmission(predictions=to_save)
        shard_proto.submission_type = wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        shard_proto.authors[:] = ['Santosh Paidi']
        shard_proto.account_name = 'santosh175@gmail.com'
        shard_proto.unique_method_name = args.model
        shard_proto.description = f'Part {part_count}'
        
        output_path = os.path.join(KAGGLE_OUTPUT_DIR, f'part{part_count}')
        with open(output_path, 'wb') as fp:
            fp.write(shard_proto.SerializeToString())
            
        part_count += 1
        del to_save
        gc.collect() 

    # Shotgun Patch & Tarball compression code remains unchanged from the notebook version.
    print("✅ Completed inference pipeline. Ready for packaging.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vib_mamba_hf')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    main(parser.parse_args())