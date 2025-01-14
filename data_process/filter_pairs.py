# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# metrics
import torch
import torchvision
import os
import numpy as np
import glob
import cv2 as cv
from dreamsim import dreamsim
from PIL import Image
import random
from tqdm import tqdm
import json
from itertools import combinations

def build_entry(img1, img2, mask, pair_num, out_dir, shuffle):
    if shuffle is True:
        if random.random() > 0.5:
            img1, img2 = img2, img1
    target = Image.open(img1)
    reference = Image.open(img2)
    target.save(os.path.join(out_dir, f"pair{pair_num:0>4}_t.png"))
    reference.save(os.path.join(out_dir, f"pair{pair_num:0>4}_r.png"))
    if mask is not None:
        mask = Image.open(mask)
        mask = np.array(mask)
        mask[mask>0] = 255
        mask = Image.fromarray(np.uint8(mask))
        masked_t = Image.composite(mask, target, mask.convert('L'))
        masked_t.save(os.path.join(out_dir, f"pair{pair_num:0>4}_tm.png"))
        mask.save(os.path.join(out_dir, f"pair{pair_num:0>4}_m.png"))
    return

class SceneDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_path, transform):
        super().__init__()
        self.img_list = sorted(glob.glob(os.path.join(root_path, "*")))
        self.transform = transform
        self.total_idx = list(combinations(range(len(img_list)),2))
        self.total_len = len(self.total_idx)
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        idx1, idx2 = self.total_idx[idx]
        img1 = preprocess(Image.open(self.img_list[idx1]))
        img2 = preprocess(Image.open(self.img_list[idx2]))
        return {"img1":img1, "img2":img2, "idx1":idx1, "idx2":idx2}
    
def collate_fn(data):
    img1 = torch.cat([example["img1"] for example in data])
    img2 = torch.cat([example["img2"] for example in data])
    idx1 = torch.tensor([example["idx1"] for example in data])
    idx2 = torch.tensor([example["idx2"] for example in data])
    return {"img1":img1, "img2":img2, "idx1":idx1, "idx2":idx2}
        
        

mode = "train"
out_dir = None
source = f"./dataset/{mode}/*"
os.makedirs(out_dir, exist_ok=True)
possible_list = []
for i, path in enumerate(sorted(glob.glob(source))):
    slen = len(glob.glob(os.path.join(path, "*")))
    possible_list.append(int(slen*(slen-1)/2))
possible_pair = sum(possible_list)
    
distance_threshold_upper = 0.2
distance_threshold_lower = 0.1
img_size = (512, 512)
batch_size = 128
dataloader_num_workers = 4
ds, preprocess = dreamsim(pretrained=True, cache_dir="./metrics/.cache")
device = "cuda"

all_scene = sorted(glob.glob(source))
dataset_json_list = []
tracking_num = 0
#https://www.geeksforgeeks.org/how-to-convert-python-dictionary-to-json/
global_pair_num = 0

with tqdm(total=possible_pair) as pbar:
    random.seed(42)
    for scene_num in range(len(all_scene)):
        scene_tag = all_scene[scene_num].split('/')[-1]
        img_list = sorted(glob.glob(os.path.join(all_scene[scene_num], "*")))
        scene_dataset = SceneDataset(all_scene[scene_num], preprocess)
        scene_dataloader = torch.utils.data.DataLoader(
            scene_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
        )
        for step, batch in enumerate(scene_dataloader):
            data_size = len(batch["idx1"])
            idx1_list = batch["idx1"]
            idx2_list = batch["idx2"]
            img1_pre = batch["img1"].to(device)
            img2_pre = batch["img2"].to(device)
            ds_dist = ds(img1_pre, img2_pre).cpu()
            pass_thres = torch.logical_and((ds_dist < distance_threshold_upper), (ds_dist > distance_threshold_lower))
            num_pass_thres = len(ds_dist[pass_thres])
            pbar.update(data_size)
            if num_pass_thres > 0:
                #print(f"paired {img1_idx} and {img2_idx} as pair num {pair}")
                for idx1_t, idx2_t in zip(idx1_list[pass_thres], idx2_list[pass_thres]):
                    img1_idx = idx1_t.item()
                    img2_idx = idx2_t.item()
                    mask1_idx = global_pair_num
                    build_entry(f"dataset/{mode}/{scene_tag}/{img_list[img1_idx].split('/')[-1]}",
                                f"dataset/{mode}/{scene_tag}/{img_list[img2_idx].split('/')[-1]}",
                                None, global_pair_num, out_dir, shuffle=True) # mask=mask_path[mask1_idx] if use, current None
                    global_pair_num += 1

                    break
            break
        #if global_pair_num >= 500:
        #    break
    print(global_pair_num)
