

import os, re, cv2, torch
import torch.nn as nn
from scipy.signal import medfilt
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# CONFIGURATION
TRAIN_DIR = r"\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\training_videos"
RAW_TEST_DIR = r"\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\testing_videos"
CLEAN_TEST_DIR = r"\Avenue_Corrected"
GK_TRAIN_DIR = r"\GK_Training_Data"
OUTPUT_PATH = r"submission.csv"
MODEL_PATH = r"Model.pth"
GK_PATH = r"Gatekeeper.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(CLEAN_TEST_DIR, exist_ok=True)

class UNet(nn.Module):
    """
    U-Net architecture for frame reconstruction. 
    Learns normal motion patterns to predict the next frame in a sequence.
    """
    def __init__(self):
        super(UNet, self).__init__()
        # Standard convolutional block
        def cb(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)
        )
        # Encoder
        self.enc1, self.enc2, self.enc3 = cb(12, 64), cb(64, 128), cb(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = cb(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec3 = cb(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec2 = cb(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec1 = cb(128, 64)
        
        # Output layer
        self.final = nn.Conv2d(64, 3, 1); self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        # Bottleneck
        b = self.bottleneck(p3)
        # Decoder path with skip connections via concatenation
        d3 = self.dec3(torch.cat((self.up3(b), e3), 1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), 1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), 1))
        return self.sigmoid(self.final(d1))

class FramePredDataset(Dataset):
    """
    Sliding window dataset that takes 4 frames as input to predict the 5th frame.
    Used for temporal anomaly detection.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = []
        video_files = {}

        # Scan directories and group frames by video ID
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    try:
                        vid_folder = os.path.basename(os.path.dirname(os.path.join(root, file)))
                        vid_id = str(int(re.findall(r'\d+', vid_folder)[-1]))                        
                        frame_num = int(re.findall(r'\d+', file)[-1])
                        if vid_id not in video_files: 
                            video_files[vid_id] = []
                        video_files[vid_id].append((frame_num, os.path.join(root, file)))
                    except: continue

        # Generate overlapping sequences within each video
        for vid_id in sorted(video_files.keys(), key=int):
            frames = sorted(video_files[vid_id], key=lambda x: x[0])
            if len(frames) < 5: continue
            for i in range(len(frames) - 4):
                self.sequences.append({
                    'inputs': [f[1] for f in frames[i : i + 4]], 
                    'target': frames[i + 4][1], 
                    'id': f"{vid_id}_{frames[i + 4][0]}"
                })

    def __len__(self): return len(self.sequences)
    
    def __getitem__(self, idx):
        item = self.sequences[idx]
        # Concatenate 4 frames along channel dimension (4*3 = 12 channels)
        inputs = torch.cat([self.transform(Image.open(p).convert('RGB')) for p in item['inputs']], dim=0)
        target = self.transform(Image.open(item['target']).convert('RGB'))
        return inputs, target, item['id']

def calculate_locality_score_enhanced(prediction, target):
    """
    Determines anomaly intensity by finding local regions with the highest reconstruction error.
    Max Pooling prevents small, fast anomalies from being washed out by background averaging.
    """
    # Absolute reconstruction error map
    err = torch.abs(prediction - target).sum(dim=1, keepdim=True)
    # Downsample via max pooling to find peak regional error
    patches = torch.nn.functional.max_pool2d(err, kernel_size=16, stride=8)
    return torch.max(patches).item()

def run_final_inference_with_padding():
    """
    Main inference pipeline with robust global post-processing for scoring accuracy.
    """
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    loader = DataLoader(FramePredDataset(CLEAN_TEST_DIR, transform=tf), batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for inp, tar, rid in tqdm(loader, desc="Detecting Anomalies"):
            pred = model(inp.to(DEVICE))
            score = calculate_locality_score_enhanced(pred, tar.to(DEVICE))
            v_p = rid[0].split('_')
            results.append({
                'Id': f"{int(v_p[0])}_{int(v_p[1])}", 
                'RawScore': score, 
                'vid': int(v_p[0]), 
                'frame': int(v_p[1])
            })

    df_raw = pd.DataFrame(results)
    
    # Align model outputs with the master list of all frames to handle missing data
    master_list = []
    for v_idx in range(1, 22):
        v_folder = str(v_idx).zfill(2)
        path = os.path.join(RAW_TEST_DIR, v_folder)
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.png')):
                    f_num = int(re.findall(r'\d+', f)[-1])
                    master_list.append({'Id': f"{v_idx}_{f_num}", 'vid': v_idx, 'frame': f_num})

    df_final = pd.merge(pd.DataFrame(master_list), df_raw[['Id', 'RawScore']], on='Id', how='left')
    df_final = df_final.sort_values(by=['vid', 'frame']).reset_index(drop=True)

    print("Executing Global Double-Normalization Pipeline")
    df_final['RawScore'] = df_final['RawScore'].ffill().bfill()

    # Step 1: Temporal smoothing using a Median Filter
    df_final['RawScore'] = medfilt(df_final['RawScore'].values, kernel_size=11)

    # Step 2: First Global Normalization to standardize the 0-1 range
    g_min1 = df_final['RawScore'].min()
    g_max1 = df_final['RawScore'].max()
    df_final['Predicted'] = (df_final['RawScore'] - g_min1) / (g_max1 - g_min1 + 1e-8)

    # Step 3: Power transformation to suppress background noise and highlight spikes
    df_final['Predicted'] = df_final['Predicted'] ** 3 

    # Step 4: Second Global Normalization to re-stretch values after power scaling
    g_min2 = df_final['Predicted'].min()
    g_max2 = df_final['Predicted'].max()
    df_final['Predicted'] = (df_final['Predicted'] - g_min2) / (g_max2 - g_min2 + 1e-8)

    # Step 5: Final Gaussian blur for a smooth anomaly probability curve
    df_final['Predicted'] = gaussian_filter1d(df_final['Predicted'].values, sigma=1.5)

    df_final[['Id', 'Predicted']].to_csv(OUTPUT_PATH, index=False)
    print(f"Global submission saved to: {OUTPUT_PATH}")


# if __name__ == "__main__":
    # 1. Prepare GK Training Data
    # prepare_gatekeeper_data(TRAIN_DIR, GK_TRAIN_DIR)

    # 2. Train Gatekeeper
    # train_gatekeeper()

    # 3. Clean the Dataset
    # run_correction_sweep()

    # 4. Train U-Net on Clean Data
    # train_unet()

    # 5. Generate Final CSV
    # run_final_inference_with_padding()

