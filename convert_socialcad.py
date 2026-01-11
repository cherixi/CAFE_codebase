import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import glob

# ================= 配置 =================
# 你刚刚转换好的 Social-CAD 根目录
DATASET_ROOT = '/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/Social-CAD_CAFE_Format'

# 输出路径
PKL_OUTPUT = 'gt_tracks_socialcad.pkl' # 生成的 pkl
TXT_OUTPUT = 'gt_tracks_socialcad.txt' # 生成的 txt (评估用)

# 图像分辨率 (用于反归一化或检查)
# 假设 Social-CAD 分辨率是 1920x1080，请务必确认！！！
IMG_WIDTH = 1920 
IMG_HEIGHT = 1080 

# 活动名称到 ID 的映射 (用于 txt)
# CAFE 示例: 'Queueing': 0, ..., 'Individual': 6, 'No': 7
# 这里的 ID 必须与你训练代码中的 ACTIVITIES 列表索引一致
ACTIVITY_MAP_NAME_TO_ID = {
    "Talking": 0,
    "Waiting": 1,
    "Walking": 2,
    "Ordering": 3,
    "Eating/Drinking": 4, # 示例
    "Unknown": 5, # or other
    # individual 不需要在这里映射，逻辑中单独处理
}

# =======================================

def main():
    pkl_data = {}
    txt_lines = []

    # 遍历所有序列文件夹 (1, 2, ... 44)
    # 假设文件夹名就是数字
    try:
        seq_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if d.isdigit()], key=int)
    except:
        seq_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])

    print(f"Found {len(seq_dirs)} sequences.")

    frame_global_counter = 0 # 用于 pkl 的 key (模拟连续帧)
    # pkl 的 key 逻辑：通常对应视频的帧索引。
    # 由于我们现在是 Clip 形式，且每个 Clip 代表一个时刻。
    # 我们假设 Key 就是 Clip 的 ID 或者 Clip 的中心帧在原视频的 ID？
    # 观察你的 cafe pkl 示例: 0, 6, 12... 间隔 6。
    # 我们的 Clip 也是离散的。为了 VideoMAE 提取方便，我们也可以用 0, 10, 20... (如果步长是10)
    # 或者简单点：如果提取脚本是按 Clip 遍历的，Key 可能是 Clip ID。
    # 这里的关键是提取脚本怎么通过 Key 找到 Box。
    
    # 决定方案：我们使用 Clip ID 作为 Key 吗？不，通常是 (video_id, frame_id) 或者全局 frame_id。
    # 让我们再看一眼 t.py 的输出: keys 是 0, 6, 12... 看起来像是在一个大视频里的帧号。
    # 鉴于 Social-CAD 是多个视频，pkl 结构可能需要是 { (vid, fid): boxes } 或者 { 'v_01_000': ... }
    # 但 CAFE 原始代码通常把所有视频拼在一起处理 VideoMAE？或者 VideoMAE 仅仅是 preprocess。
    
    # **重要假设**：为了兼容性，我们将 Key 设为 (video_id * 100000 + frame_id) 这种唯一 ID，
    # 或者如果 VideoMAE 脚本仅仅用于当前数据集，我们按读取顺序生成 Key。
    # 但为了最稳妥，我们先生成 txt，pkl 的用途通常是辅助。
    # 既然你给了 pkl 示例是 0, 6, 12，这明显是针对单个长视频或者拼接视频的相对时间戳。
    # 我们这里生成一个 { (video_id, clip_id): array } 的映射可能更安全，需看你提取特征的代码。
    
    # 修改策略：既然你只有 pkl 的内容没有提取代码，不管是啥 Key，只要提取代码能对应上就行。
    # 观察 cafe 的 pkl Key 是连续增长的，我们这里也用一个全局增长的 Key 模拟。
    
    global_pkl_key = 0 
    
    for seq_name in tqdm(seq_dirs):
        seq_path = os.path.join(DATASET_ROOT, seq_name)
        seq_id = int(seq_name)
        
        # 获取该序列下所有 clip (0, 1, 2...)
        clip_dirs = sorted([d for d in os.listdir(seq_path) if d.isdigit()], key=int)
        
        for clip_name in clip_dirs:
            clip_path = os.path.join(seq_path, clip_name)
            ann_path = os.path.join(clip_path, 'ann.json')
            
            if not os.path.exists(ann_path):
                continue
                
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            
            # 解析 ann.json
            figures = ann.get('figures', [])
            
            # 准备 pkl 数据: [N, 5] (tid, cx, cy, w, h)
            pkl_boxes = []
            
            # 准备 txt 数据
            # 需要知道这个 Clip 对应的真实帧 ID
            # 我们在转换时把真实帧 ID 丢了吗？
            # 这是一个问题！转换脚本生成的文件夹名是 0, 1, 2...
            # 我们需要找回真实的 Frame ID。
            # 笨办法：去读 frames_0.jpg 的原始文件名？不，已经被重命名了。
            # 聪明办法：在 ann.json 里并没有存真实帧 ID...
            # **修正**: 回去修改转换脚本增加真实帧 ID 字段是最完美的。
            # **临时办法**: 假设 Clip 0 对应原数据的第一个标注帧...
            # 我们假设 Clip 0 -> 标注列表第 0 项。
            
            # 为了继续，我们假设你有一个映射或者从文件名推断。
            # 实际上，txt 需要真实帧 ID (例如 11, 21...)。
            # 我们可以通过 seq_id 和 clip_id (0, 1...) 反推，前提是知道步长。
            # 假设步长是 10，起始是 11。 Fid = 11 + clip_id * 10
            estimated_real_fid = 11 + int(clip_name) * 10 
            # (请根据实际切分逻辑调整这个公式！！！)

            for fig in figures:
                tid = float(fig['id'])
                label = fig['label'] # groupX 或 individual
                
                # 获取 Box (关键帧)
                shape = fig['shapes'][0] # 假设只有一个 shape (keyframe)
                coords = shape['coordinates']
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                
                # 1. 构建 pkl entry (归一化 cx, cy, w, h)
                cx = (x1 + x2) / 2.0 / IMG_WIDTH
                cy = (y1 + y2) / 2.0 / IMG_HEIGHT
                w = (x2 - x1) / IMG_WIDTH
                h = (y2 - y1) / IMG_HEIGHT
                pkl_boxes.append([tid, cx, cy, w, h])
                
                # 2. 构建 txt entry
                # 格式: video_id, person_id, frame_id, x1, y1, x2, y2, group_id, activity_id
                
                # 解析 Group ID 和 Activity ID
                if label == 'individual':
                    gid = -1
                    # 对于 individual，activity 可能是 'Individual' 类
                    act_id = len(ACTIVITY_MAP_NAME_TO_ID) # 假设最后一个是 Individual
                    # 或者从 attributes 里读
                else:
                    # label="group1" -> gid=1
                    gid = int(label.replace('group', ''))
                    
                    # 获取 Activity
                    act_key = fig['attributes'][0]['value']['key']
                    act_id = ACTIVITY_MAP_NAME_TO_ID.get(act_key, -1)
                
                # 写入 txt 行
                line = f"{seq_id} {int(tid)} {estimated_real_fid} {int(x1)} {int(y1)} {int(x2)} {int(y2)} {gid} {act_id}\n"
                txt_lines.append(line)
            
            # 保存 pkl
            pkl_data[global_pkl_key] = np.array(pkl_boxes, dtype=np.float32)
            global_pkl_key += 6 # 模拟 cafe 的步长，保持一致性

    # 保存
    print(f"Writing pickle to {PKL_OUTPUT}...")
    with open(PKL_OUTPUT, 'wb') as f:
        pickle.dump(pkl_data, f)
        
    print(f"Writing txt to {TXT_OUTPUT}...")
    with open(TXT_OUTPUT, 'w') as f:
        f.writelines(txt_lines)

    print("Done.")

if __name__ == '__main__':
    main()