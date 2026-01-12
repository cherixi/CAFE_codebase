import os
import json
import pickle
import shutil
# import cv2
import numpy as np
import glob
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
# Adjust these paths to your environment

# Path containing sequence folders with images (e.g. .../1/images/*.jpg)
IMAGES_ROOT = r"D:\ActivityDataset"

# Path containing annotation files (e.g. .../1/annotations.txt or .../1.txt)
ANNOTATIONS_ROOT = r"D:\Social-human-activity-understanding-and-grouping-master\Social-human-activity-understanding-and-grouping-master\social_CAD\social_CAD"

DEST_ROOT = r"D:\cafe_social_cad"
IMG_WIDTH = 720
IMG_HEIGHT = 480

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
    print(f"Starting conversion...")
    print(f"Images Root: {IMAGES_ROOT}")
    print(f"Annotations Root: {ANNOTATIONS_ROOT}")
    
    # --- DIAGNOSTIC BLOCK ---
    if os.path.exists(IMAGES_ROOT):
        items = os.listdir(IMAGES_ROOT)
        print(f"[DEBUG] IMAGES_ROOT contains {len(items)} items. First 5: {items[:5]}")
    else:
        print(f"[ERROR] IMAGES_ROOT does not exist: {IMAGES_ROOT}")
        return

    if os.path.exists(ANNOTATIONS_ROOT):
        items = os.listdir(ANNOTATIONS_ROOT)
        print(f"[DEBUG] ANNOTATIONS_ROOT contains {len(items)} items. First 5: {items[:5]}")
    else:
        print(f"[ERROR] ANNOTATIONS_ROOT does not exist: {ANNOTATIONS_ROOT}")
        return
    # ------------------------

    ensure_dir(DEST_ROOT)

    global_tracks = {} # Key: (vid, clip), Value: {5: np.array of shape (N, 5)}
    
    # Placeholder for collecting all clips metadata for gt_tracks.txt
    all_tracks_txt_lines = []

    for seq_id in tqdm(ALL_SEQS, desc="Processing Sequences"):
        # -----------------------------------------------------------
        # RESOLVE PATHS
        # -----------------------------------------------------------
        
        # 1. Image Directory Logic
        # User specified: IMAGES_ROOT/seq01/ (Images directly inside)
        # Note: ALL_SEQS are strings "1"..."44"
        
        # Attempt: IMAGES_ROOT/seq01/
        img_folder_name = f"seq{seq_id.zfill(2)}"
        img_seq_dir = os.path.join(IMAGES_ROOT, img_folder_name)
        
        if not os.path.exists(img_seq_dir):
             # Fallback: maybe just 1, or 01
            if os.path.exists(os.path.join(IMAGES_ROOT, seq_id)):
                img_seq_dir = os.path.join(IMAGES_ROOT, seq_id)
            elif os.path.exists(os.path.join(IMAGES_ROOT, seq_id.zfill(2))):
                img_seq_dir = os.path.join(IMAGES_ROOT, seq_id.zfill(2))

        # 2. Annotation File Logic
        # User specified: ANNOTATIONS_ROOT/1_annotations.txt
        ann_filename = f"{seq_id}_annotations.txt"
        annotations_file = os.path.join(ANNOTATIONS_ROOT, ann_filename)
        
        if not os.path.exists(annotations_file):
             # Fallback: Try with padding 01_annotations.txt
            annotations_file = os.path.join(ANNOTATIONS_ROOT, f"{seq_id.zfill(2)}_annotations.txt")
            
        if not os.path.exists(annotations_file):
             # Fallback: Try generic annotations.txt inside folder if exists, or just {id}.txt
             # But prioritizing user request
            pass

        # Check validity
        if not os.path.exists(img_seq_dir):
            print(f"Warning: Images for sequence {seq_id} not found at {img_seq_dir}")
            continue
        if not os.path.exists(annotations_file):
            print(f"Warning: Annotations for sequence {seq_id} not found at {annotations_file}")
            continue

        # DEST STRUCTURE: DEST_ROOT/seq_id/clip_id/
        ensure_dir(os.path.join(DEST_ROOT, seq_id))

        seq_data = load_socialcad_annotations(annotations_file) 
        # [DEBUG]
        print(f"[DEBUG] Loaded {len(seq_data)} frames for sequence {seq_id}")
        if len(seq_data) == 0:
            print(f"[WARNING] No valid data parsed for seq {seq_id}. Check load_socialcad_annotations logic.")
        else:
            first_key = list(seq_data.keys())[0]
            print(f"[DEBUG] Seq {seq_id} Sample Frame {first_key}: {seq_data[first_key]}")

        # Determine clips
        # CAFE strategy: sliding window? Or just centered on annotated frames?
        # User said: "Window: t is frame. Clip [t-5, t+4]. Total 10 frames."
        # We iterate over every frame 't' that has annotations.
        
        annotated_frames = sorted(seq_data.keys())
        
        clip_counter = 0
        
        for t in annotated_frames:
            # t is the "key frame"
            # Frames to extract: t-5 to t+4

            # [FIX] Boundary check: start_frame must be >= 1
            if t - 5 < 1:
                continue

            start_frame = t - 5
            end_frame = t + 5 # exclusive for range, so t+4 is last
            
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
            missing_frames = [] # Debug
            for i, f_idx in enumerate(range(start_frame, end_frame)):
                # Source Image Name logic
                # Try various common formats
                src_img_paths_to_try = [
                    os.path.join(img_seq_dir, f"frame{f_idx:04d}.jpg"), # User confirmed: frame0001.jpg
                    os.path.join(img_seq_dir, f"{f_idx:05d}.jpg"),
                    os.path.join(img_seq_dir, f"{f_idx:04d}.jpg"),
                    os.path.join(img_seq_dir, f"{f_idx:06d}.jpg"),
                    os.path.join(img_seq_dir, f"{f_idx}.jpg"),
                    os.path.join(img_seq_dir, f"frame{f_idx:05d}.jpg"),
                    os.path.join(img_seq_dir, f"frame_{f_idx:05d}.jpg"),
                    os.path.join(img_seq_dir, f"{f_idx:05d}.png"),
                    os.path.join(img_seq_dir, f"{f_idx:04d}.png")
                ]
                
                found_src = None
                for p in src_img_paths_to_try:
                    if os.path.exists(p):
                        found_src = p
                        break
                
                if found_src is None:
                    valid_clip = False
                    missing_frames.append(f_idx)
                    # [DEBUG] Silent skip for missing frames (common at end of videos)
                    break
                    
                dst_img_name = f"frames_{i}.jpg" # CAFE expects frames_0.jpg ... frames_9.jpg
                dst_img_path = os.path.join(images_dir, dst_img_name)
                
                if not os.path.exists(dst_img_path):
                    shutil.copy(found_src, dst_img_path)

            if not valid_clip:
                # Cleanup if failed
                if os.path.exists(clip_dir):
                    shutil.rmtree(clip_dir)
                # [DEBUG]
                # print(f"[DEBUG] Clip {clip_id} invalid. Missing frames: {missing_frames}")
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
                
                # CAFE convention: Individual = -1, Groups = 0, 1, 2...
                # Updated logic: if group size <= 1, FORCE gid to -1 for txt output
                if group_counts.get(gid, 0) <= 1:
                    gid = -1
                
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
        print(f"[ERROR] File not found: {file_path}")
        return {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # [DEBUG]
        if len(lines) > 0:
            print(f"[DEBUG] Processing {os.path.basename(file_path)}: {len(lines)} lines. First line: {lines[0].strip()}")
        else:
            print(f"[DEBUG] File is empty: {file_path}")

        for line in lines:
            parts = line.strip().split('\t')
            # If split by tab fails (len < 6), try space/comma just in case
            if len(parts) < 6:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    parts = line.strip().split()
            
            if len(parts) < 9: 
                # [DEBUG]
                # print(f"Skipping line: insufficient parts {len(parts)}: {line.strip()}")
                continue
            
            try:
                # Inferred mapping:
                # 1	339	191	422	356	5	4	3	1
                
                # IMPORTANT: In file 1_annotations.txt, first col is '1'. 
                # Is '1' the frame ID? Usually datasets index frame 1..N.
                # Or is '1' the Track ID? 
                
                # Let's verify against logic:
                # The user's sample output confirms lines like: "1 339 ..."
                # If column 0 is frame_idx, then Frame 1 has annotations.
                
                frame_idx = int(parts[0])
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                
                # Check for empty string or parse errors specifically
                if str(parts[5]).strip() == '': activity = 1 
                else: activity = int(parts[5])
                
                track_id = int(parts[7])
                
                group_id_raw = int(parts[8])
                group_id = group_id_raw - 1 if group_id_raw > 0 else 0 
                
                if frame_idx not in data_map:
                    data_map[frame_idx] = []
                    
                data_map[frame_idx].append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'activity': activity,
                    'group_id': group_id # Now 0-based
                })
            except ValueError as e:
                # [DEBUG]
                # print(f"ValueError parsing line: {line.strip()} - {e}")
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
        # Logic update: group_id is now 0-based.
        # But we need to distinguish "Individual" (no group) from "Group 0".
        # If original file had correct grouping, maybe we should just treat filtered individuals later.
        
        # User requested "Group ID start from 0".
        # In current logic: group_counts keys are now 0, 1, 2...
        
        # Check if this group ID has > 1 member
        if group_counts[gid] > 1:
            # [FIX 2] User requested JSON group label start from 1
            # Current gid is 0-based, so we output group{gid+1}
            label = f"group{gid + 1}"
        else:
            # If it's a single person group, CAFE treats as individual
            label = "individual"
            # And for gt_tracks.txt logic later, we might want to set gid to -1
            
        # Structure for figure
        
        # [FIX 3] Clip coordinates to 0
        b_x1 = max(0.0, bbox[0])
        b_y1 = max(0.0, bbox[1])
        b_x2 = max(0.0, bbox[2])
        b_y2 = max(0.0, bbox[3])
        
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
                    "keyframe": True, # [FIX 1] Added keyframe: true
                    "type": "rect", # Assumption
                    "frame": 5, # We are centering at frame 5 (0-9 range, 5 is index 5 or 4? User said t is annotated. Let's say index 5)
                    "coordinates": [
                        [b_x1, b_y1],
                        [b_x2, b_y2]
                    ]
                }
            ]
        }
        j['figures'].append(figure)
        
    return j

if __name__ == "__main__":
    convert_socialcad_to_cafe()
