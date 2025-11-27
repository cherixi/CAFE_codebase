import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Extract VideoMAE features')
    parser.add_argument('--data_path', default='../Dataset/', type=str, help='data path')
    parser.add_argument('--dataset', default='cafe', type=str, help='dataset name')
    parser.add_argument('--save_dir', default='./videomae_features', type=str, help='directory to save features')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    return parser.parse_args()

def load_frames(clip_path, num_frames=16):
    image_dir = os.path.join(clip_path, 'images')
    if not os.path.exists(image_dir):
        return None
    
    # Filter for image files
    frame_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Sort by frame number if possible (assuming format frames_xxx.jpg)
    try:
        frame_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except:
        pass # Fallback to string sort if filename format is different

    total_frames = len(frame_files)
    
    if total_frames == 0:
        return None

    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for i in indices:
        frame_path = os.path.join(image_dir, frame_files[i])
        try:
            img = Image.open(frame_path).convert('RGB')
            frames.append(img)
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
            return None
            
    return frames

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading VideoMAE model to {device}...")
    model_name = "MCG-NJU/videomae-base"
    try:
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Failed to load model from huggingface: {e}")
        print("Please ensure you have internet connection or pre-downloaded weights.")
        return

    model.eval()
    
    # Construct dataset root path
    # Based on dataloader logic: data_path/dataset_name
    dataset_root = os.path.join(args.data_path, args.dataset)
    if not os.path.exists(dataset_root):
        print(f"Dataset path not found: {dataset_root}")
        # Try without dataset name if user provided full path
        if os.path.exists(args.data_path):
             dataset_root = args.data_path
        else:
             return

    save_root = args.save_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        
    print(f"Scanning dataset at {dataset_root}...")
    
    tasks = []
    
    # Traverse dataset structure: dataset_root/vid/cid
    # We expect directories like '1', '2', etc. for videos
    vids = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    
    for vid in vids:
        vid_path = os.path.join(dataset_root, vid)
        # Check if it's a video directory (should contain clip directories)
        cids = sorted([d for d in os.listdir(vid_path) if os.path.isdir(os.path.join(vid_path, d))])
        
        for cid in cids:
            clip_path = os.path.join(vid_path, cid)
            # Check if it contains images folder
            if not os.path.exists(os.path.join(clip_path, 'images')):
                continue
                
            save_name = f"{vid}_{cid}.npy"
            save_path = os.path.join(save_root, save_name)
            
            if not os.path.exists(save_path):
                tasks.append((clip_path, save_path))
    
    print(f"Found {len(tasks)} clips to process.")
    
    for clip_path, save_path in tqdm(tasks):
        frames = load_frames(clip_path, num_frames=16)
        if frames is None:
            print(f"Skipping {clip_path} (no frames found)")
            continue
            
        inputs = processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            # Extract CLS token: [1, 768]
            feat = last_hidden_state[:, 0, :].cpu().numpy()
            
        np.save(save_path, feat)
        
    print("Extraction complete.")

if __name__ == '__main__':
    main()
