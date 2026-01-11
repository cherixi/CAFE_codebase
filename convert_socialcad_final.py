import os
import json
import pickle
import shutil
import cv2
import numpy as np
import glob
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
# Adjust these paths to your environment
SOURCE_ROOT = r"D:\Social-CAD\data"  # Path containing sequence folders (e.g., seq_01, seq_02...)
DEST_ROOT = r"D:\Cafe_Dataset\Cafe_Dataset\Dataset\cafe_social_cad"
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Social-CAD Activity Mapping
# 1:NA, 2:Crossing, 3:Waiting, 4:Queueing, 5:Walking, 6:Talking
ACTIVITY_MAP = {
    1: 'NA',
    2: 'Crossing',
    3: 'Waiting',
    4: 'Queueing',
    5: 'Walking',
    6: 'Talking'
}

# CAFE Class IDs (Must match train.py order)
# ACTIVITIES = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking', 'Individual', 'No']
# Indices: NA=0, Crossing=1, ...
# Wait, based on previous turn, I replaced ACTIVITIES in train.py:
# ACTIVITIES = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking', 'Individual', 'No']
# So the mapping should be:
NAME_TO_CAFE_ID = {
    'NA': 0,
    'Crossing': 1,
    'Waiting': 2,
    'Queueing': 3,
    'Walking': 4,
    'Talking': 5
    # Individual and No are special handling
}

# CAFE Dataloader expects these keys in ann.json
# We will write the activity name strings. 
# NOTE: You must update `dataloader/cafe.py` ACTIVITIES list to match these names!

# Train/Test Split (Social-CAD usually has 44 sequences)
# We'll use all for training or split them. Let's assume passed in lists or all.
ALL_SEQS = [str(i) for i in range(1, 45)] # 1 to 44

# =============================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_box(x1, y1, x2, y2, w_img, h_img):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [
        cx / w_img,
        cy / h_img,
        w / w_img,
        h / h_img
    ]

def convert_socialcad_to_cafe():
    print(f"Starting conversion from {SOURCE_ROOT} to {DEST_ROOT}")
    ensure_dir(DEST_ROOT)

    global_tracks = {} # Key: (vid, clip), Value: {5: np.array of shape (N, 5)}
    
    # Iterate over sequences
    # Expecting structure: SOURCE_ROOT/1/images/... and SOURCE_ROOT/1/annotations.txt (or consistent format)
    # We'll assume the user has a loader or we parse a standard format.
    # Since I don't have the explicit Social-CAD loader logic from the user, 
    # I will provide a placeholder frame loop that user needs to adapt to their *specific* source files.
    
    # Placeholder for collecting all clips metadata for gt_tracks.txt
    all_tracks_txt_lines = []

    for seq_id in tqdm(ALL_SEQS, desc="Processing Sequences"):
        seq_path = os.path.join(SOURCE_ROOT, seq_id)
        # Check if seq exists (handle padding difference e.g., '01' vs '1')
        if not os.path.exists(seq_path):
            # Try zero padding
            seq_path = os.path.join(SOURCE_ROOT, seq_id.zfill(2))
        
        if not os.path.exists(seq_path):
            print(f"Warning: Sequence {seq_id} not found at {seq_path}")
            continue

        # DEST STRUCTURE: DEST_ROOT/seq_id/clip_id/
        ensure_dir(os.path.join(DEST_ROOT, seq_id))

        # LOAD ANNOTATIONS FOR SEQUENCE
        # Assuming annotations contain: frame_id, track_id, x1, y1, x2, y2, activity_id, group_id
        # You need to implement `load_annotations` based on your raw file format
        # For now, I'll simulate or assume a dict: { frame_idx: [ {track_id, bbox, act, group}, ... ] }
        annotations_file = os.path.join(seq_path, 'annotations.txt') # adjust name
        if not os.path.exists(annotations_file):
             # Try other common names
             annotations_file = os.path.join(seq_path, 'annotations.xml') 

        seq_data = load_socialcad_annotations(annotations_file) 
        
        # Determine clips
        # CAFE strategy: sliding window? Or just centered on annotated frames?
        # User said: "Window: t is frame. Clip [t-5, t+4]. Total 10 frames."
        # We iterate over every frame 't' that has annotations.
        
        annotated_frames = sorted(seq_data.keys())
        
        clip_counter = 0
        
        for t in annotated_frames:
            # t is the "key frame"
            # Frames to extract: t-5 to t+4
            start_frame = t - 5
            end_frame = t + 5 # exclusive for range, so t+4 is last
            
            # Check bounds (assuming images exist)
            # We assume images are named frame0001.jpg etc. or similar.
            # check availability
            
            # Create Clip Folder
            clip_id = str(clip_counter)
            clip_dir = os.path.join(DEST_ROOT, seq_id, clip_id)
            images_dir = os.path.join(clip_dir, 'images')
            try:
                ensure_dir(images_dir)
            except OSError:
                pass # multiple threads?

            # Copy Images
            valid_clip = True
            for i, f_idx in enumerate(range(start_frame, end_frame)):
                # Source Image Name logic
                src_img_name = f"{f_idx:05d}.jpg" # SocialCAD naming? usually 0-padded
                # Try finding file
                src_img_path = os.path.join(seq_path, 'images', src_img_name) 
                
                if not os.path.exists(src_img_path):
                     # Try 4 digits
                    src_img_path = os.path.join(seq_path, 'images', f"{f_idx:04d}.jpg")
                
                if not os.path.exists(src_img_path):
                    valid_clip = False
                    break
                    
                dst_img_name = f"frames_{i}.jpg" # CAFE expects frames_0.jpg ... frames_9.jpg
                dst_img_path = os.path.join(images_dir, dst_img_name)
                
                if not os.path.exists(dst_img_path):
                    shutil.copy(src_img_path, dst_img_path)

            if not valid_clip:
                # Cleanup if failed
                if os.path.exists(clip_dir):
                    shutil.rmtree(clip_dir)
                continue

            # Generate ann.json
            frame_data = seq_data[t]
            ann_json = create_ann_json(frame_data)
            
            with open(os.path.join(clip_dir, 'ann.json'), 'w') as f:
                json.dump(ann_json, f, indent=4)
                
            # Accumulate Track Data for PKL
            # For each person in THIS frame 't'
            # Format: [track_id, cx_norm, cy_norm, w_norm, h_norm]
            clip_tracks = []
            
            # --- Pre-calculate group size for label logic ---
            group_counts = {}
            for ent in frame_data:
                g = ent['group_id']
                group_counts[g] = group_counts.get(g, 0) + 1
            # ------------------------------------------------

            for entity in frame_data:
                pid = float(entity['track_id'])
                x1, y1, x2, y2 = entity['bbox']
                norm_box = normalize_box(x1, y1, x2, y2, IMG_WIDTH, IMG_HEIGHT)
                
                # Append [pid, cx, cy, w, h]
                clip_tracks.append([pid] + norm_box)
                
                # --- GENERATE GT_TRACKS.TXT LINE ---
                # Format from CAFE eval: 
                # row[0-2]: vid, cid, fid
                # row[3-6]: x1, y1, x2, y2
                # row[7]: group_id
                # row[8]: activity_id
                # row[9]: group_score (pred only)
                # row[10]: actor_score (pred only)
                
                # frame_id is fixed to 5 (the keyframe inside the clip) in CAFE dataset clips
                
                act_str = ACTIVITY_MAP.get(entity['activity'], 'NA')
                cafe_act_id = NAME_TO_CAFE_ID.get(act_str, 7) # Default to 7 (No) if unknown, or 0 (NA)
                
                gid = entity['group_id']
                
                # Ensure seq_id is int for txt format
                try:
                    s_id_int = int(seq_id)
                except ValueError:
                    s_id_int = -1 

                # x1 y1 x2 y2 are raw pixels
                txt_line = f"{s_id_int} {clip_id} 5 {int(x1)} {int(y1)} {int(x2)} {int(y2)} {gid} {cafe_act_id}\n"
                all_tracks_txt_lines.append(txt_line)
                # -----------------------------------
            
            if len(clip_tracks) > 0:
                # Key: (video_bin_name, clip_bin_name) e.g., ("1", "0")
                key = (seq_id, clip_id)
                global_tracks[key] = {5: np.array(clip_tracks, dtype=np.float32)}

            clip_counter += 1
            
    # Save Global Files
    print("Saving gt_tracks.pkl...")
    with open(os.path.join(DEST_ROOT, 'gt_tracks.pkl'), 'wb') as f:
        pickle.dump(global_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving gt_tracks.txt with {len(all_tracks_txt_lines)} lines...")
    with open(os.path.join(DEST_ROOT, 'gt_tracks.txt'), 'w') as f:
        f.writelines(all_tracks_txt_lines)
        
    print("Done!")

def load_socialcad_annotations(file_path):
    """
    Parses Social-CAD annotation file.
    Expects Tab-separated values based on user sample.
    Format inferred from sample "1 339 191 422 356 5 4 3 1":
    Col 0: Frame ID
    Col 1: x1
    Col 2: y1
    Col 3: x2
    Col 4: y2
    Col 5: Individual Action (1-6) -> Mapped to ACTIVITY_MAP
    Col 6: Group Action (Int)
    Col 7: Track ID (Int)
    Col 8: Group ID (Int)
    """
    data_map = {}
    
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            # If split by tab fails (len < 6), try space/comma just in case
            if len(parts) < 6:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    parts = line.strip().split()
            
            if len(parts) < 9: continue
            
            try:
                # Inferred mapping:
                # 1	339	191	422	356	5	4	3	1
                frame_idx = int(parts[0])
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                
                activity = int(parts[5]) 
                # parts[6] is Group Action, ignored for now as we map Ind Action to Group Action logic later if needed
                # or strictly use parts[5] as per ACTIVITY_MAP
                
                track_id = int(parts[7])
                group_id = int(parts[8])
                
                if frame_idx not in data_map:
                    data_map[frame_idx] = []
                    
                data_map[frame_idx].append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'activity': activity,
                    'group_id': group_id
                })
            except ValueError:
                continue
                
    return data_map

def create_ann_json(frame_entities):
    """
    Creates the complex dict structure for ann.json
    """
    
    j = {
        "framesCount": 10,
        "framesEach": 1,
        "figures": []
    }
    
    for ent in frame_entities:
        act_id = ent['activity'] # 1-6
        grp_id = ent['group_id']
        
        # Calculate Group Size to decide label
        # Simple logic: count members of this group in this frame?
        # User Logic: If group_size > 1 -> "groupX", else "individual"
        
        # Pre-calc group sizes for this frame
        # In efficient implementation, do this outside loop
        pass

    # Re-loop to fill figures
    # First, let's count group sizes
    group_counts = {}
    for ent in frame_entities:
        g = ent['group_id']
        if g not in group_counts: group_counts[g] = 0
        group_counts[g] += 1
        
    for ent in frame_entities:
        tid = ent['track_id']
        bbox = ent['bbox'] # [x1, y1, x2, y2]
        act_id = ent['activity']
        gid = ent['group_id']
        
        act_name = ACTIVITY_MAP.get(act_id, 'NA')
        
        # Determine Label
        if gid > 0 and group_counts[gid] > 1:
            label = f"group{gid}"
        else:
            label = "individual"
            
        # Structure for figure
        figure = {
            "id": tid,
            "label": label,
            "attributes": [
                {
                    "name": "activity", # Assumption
                    "value": {
                        "key": act_name
                    }
                }
            ],
            "shapes": [
                {
                    "type": "rect", # Assumption
                    "frame": 5, # We are centering at frame 5 (0-9 range, 5 is index 5 or 4? User said t is annotated. Let's say index 5)
                    "coordinates": [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[3]]
                    ]
                }
            ]
        }
        j['figures'].append(figure)
        
    return j

if __name__ == "__main__":
    convert_socialcad_to_cafe()
