import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='Extract VideoMAE V2 Giant features')
    parser.add_argument('--data_path', default='../Dataset/', type=str, help='Root path of dataset')
    parser.add_argument('--dataset', default='cafe', type=str, help='Dataset name (cafe)')
    parser.add_argument('--save_dir', default='./videomae_features_giant', type=str, help='Directory to save .npy files')
    parser.add_argument('--device', default='cuda', type=str, help='Device (cuda or cpu)')
    return parser.parse_args()

def load_frames(clip_path, num_frames=16):
    """
    读取视频片段的帧，并均匀采样 16 帧。
    """
    image_dir = os.path.join(clip_path, 'images')
    if not os.path.exists(image_dir):
        # 兼容性调整：如果 images 文件夹不存在，尝试直接在 clip_path 下寻找图片
        image_dir = clip_path
    
    # 1. 获取所有图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(image_dir):
        return None
        
    frame_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    
    if len(frame_files) == 0:
        return None
        
    # 2. 排序 (防止帧序错乱)
    try:
        frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    except:
        frame_files.sort()

    # 3. 均匀采样 16 帧
    total_frames = len(frame_files)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames = []
    for i in indices:
        img_path = os.path.join(image_dir, frame_files[i])
        try:
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return None
            
    return frames

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. 加载 VideoMAE V2 Giant 模型
    # ---------------------------------------------------------
    # 注意：V2 Giant 模型通常由 OpenGVLab 托管
    model_name = "OpenGVLab/VideoMAEv2-giant" 
    print(f"Loading model: {model_name} ...")
    print("(This may take a while to download ~5GB weights for the first time)")
    
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        print("Tip: Ensure you have 'transformers' and 'protobuf' installed.")
        return

    # 2. 准备保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created save directory: {args.save_dir}")

    # 3. 确定数据根目录
    root_path = args.data_path
    if args.dataset in os.listdir(root_path):
        root_path = os.path.join(root_path, args.dataset)
        
    print(f"Scanning data from: {root_path}")
    
    if not os.path.exists(root_path):
        print(f"Error: Data path {root_path} does not exist!")
        return

    video_list = sorted(os.listdir(root_path))
    print(f"Found {len(video_list)} videos.")

    count = 0
    skip_count = 0
    
    # 4. 主循环
    # 过滤掉非视频文件夹 (根据 CAFE 数据集结构，视频文件夹通常是数字或特定命名)
    # 这里简单过滤掉代码库常见的文件夹
    ignore_folders = ['result', 'outputs', 'models', 'scripts', 'util', 'dataloader', 'evaluation', 'videomae_features', 'videomae_features_giant', '__pycache__', '.git', '.vscode']
    
    for vid in tqdm(video_list, desc="Processing Videos"):
        if vid in ignore_folders or vid.startswith('.'):
            continue
            
        video_path = os.path.join(root_path, vid)
        if not os.path.isdir(video_path):
            continue
            
        clip_list = sorted(os.listdir(video_path))
        for cid in clip_list:
            clip_path = os.path.join(video_path, cid)
            if not os.path.isdir(clip_path):
                continue
                
            # 构造保存文件名: 1_0.npy (video_clip.npy)
            save_name = f"{vid}_{cid}.npy"
            save_path = os.path.join(args.save_dir, save_name)
            
            # 如果已存在，跳过
            if os.path.exists(save_path):
                continue
            
            # 加载帧
            frames = load_frames(clip_path, num_frames=16)
            if frames is None or len(frames) != 16:
                skip_count += 1
                continue
                
            # 推理
            try:
                inputs = processor(list(frames), return_tensors="pt")
                # B, T, C, H, W -> B, C, T, H, W (VideoMAEv2-giant 需要这个维度变换)
                # processor 输出的 pixel_values 默认是 [B, T, C, H, W]
                # 但 OpenGVLab 的实现可能需要 [B, C, T, H, W]
                # 让我们根据官方示例进行 permute
                inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
                # 注意：VideoMAEImageProcessor 通常输出 [B, T, C, H, W]
                # 如果模型报错维度问题，请取消上面这行的注释
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # 提取 CLS token
                    # Giant 的维度通常是 1408
                    
                    if isinstance(outputs, torch.Tensor):
                        last_hidden_state = outputs
                    elif hasattr(outputs, 'last_hidden_state'):
                        last_hidden_state = outputs.last_hidden_state
                    else:
                        # 如果是 tuple 或其他结构，尝试获取第一个元素
                        last_hidden_state = outputs[0]
                    
                    # 维度检查与处理
                    if last_hidden_state.dim() == 3:
                        # [Batch, SeqLen, Hidden] -> 取 [0, 0, :] 即第一个样本的 CLS token
                        feat = last_hidden_state[0, 0, :].cpu().numpy()
                    elif last_hidden_state.dim() == 2:
                        # 可能是 [Batch, Hidden] (已经池化过)
                        feat = last_hidden_state[0, :].cpu().numpy()
                    else:
                        raise ValueError(f"Unexpected output shape: {last_hidden_state.shape}")
                
                # 保存
                np.save(save_path, feat)
                count += 1
                
                # 显存清理 (Giant 模型显存占用大，定期清理是个好习惯)
                if count % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {vid}/{cid}: {e}")
                skip_count += 1

    print(f"\nExtraction finished!")
    print(f"Successfully processed: {count} clips")
    print(f"Skipped: {skip_count} clips")
    print(f"Features saved to: {args.save_dir}")
    if 'feat' in locals():
        print(f"Feature Dimension: {feat.shape}")

if __name__ == '__main__':
    main()
