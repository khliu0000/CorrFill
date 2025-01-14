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

dest = None
source = "MegaDepth/phoenix/S6/zl548/MegaDepth_v1/*" 
img_size = (512, 512)

dataset_json = []

all_scene = sorted(glob.glob("MegaDepth/phoenix/S6/zl548/MegaDepth_v1/*" ))
f = open("MegaDepth/newset_resize_log.txt", "w")

with tqdm(total=len(all_scene)) as pbar:
    for scene_num in range(len(all_scene)):
        img_list = sorted(glob.glob(os.path.join(all_scene[scene_num], "dense*/imgs/*")))
        scene_img = len(img_list)
        os.makedirs(os.path.join(dest, f"{all_scene[scene_num].split('/')[-1]}/"), exist_ok=True)
        for img_num in range(scene_img):
            img = Image.open(img_list[img_num])
            size = img.size
            short_side = np.min((size[0], size[1]))
            offset_x, offset_y = 0, 0
            if size[0]>short_side:
                offset_x = random.randrange(0, size[0]-short_side)
            elif size[1]>short_side:
                offset_y = random.randrange(0, size[1]-short_side)
            
            img_box = (offset_x, offset_y, offset_x+short_side, offset_y+short_side)
            img = img.crop(img_box)
            img = img.resize(img_size, Image.BICUBIC)
        
            saveto = os.path.join(dest, f"{all_scene[scene_num].split('/')[-1]}/{img_list[img_num].split('/')[-1]}")
            img.save(saveto)
        pbar.update(1)
