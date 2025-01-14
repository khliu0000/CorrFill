# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import numpy as np
import random
import os
import torch
import shutil
import glob

number_of_test_image = 1000
total_scene = 195

source = "MegaDepth/newset512/*"
possible_list = []
for i, path in enumerate(sorted(glob.glob(source))):
    slen = len(glob.glob(os.path.join(path, "*")))
    possible_list.append(int(slen*(slen-1)/2))

jsondata = json.load(open("MegaDepth/512_all.json"))  # Files that store pairs of image filtered by DreamSim

pairnum_dict = json.load(open("MegaDepth/scene_pair_num.json"))

random.seed(42)
sampled_idx = random.choices(range(195), k=number_of_test_image)
sampled_idx, idx_count = np.unique(sampled_idx, return_counts=True)
test_set_entries = []
acc_num = 0
for i, ele in enumerate(pairnum_dict.items()):
    if i in sampled_idx:
        count = idx_count[np.where(sampled_idx == i)]
        idxs = random.sample(range(ele[1]), int(count))
        for idx in idxs:
            test_set_entries.append(jsondata[acc_num + idx])
    acc_num += ele[1]
with open("MegaDepth/test1000_entries.json", "w") as outfile: 
    json.dump(test_set_entries, outfile, indent=4)
    
dest_dir = "test_set/validation500pairs/megadepth/"
os.makedirs(dest_dir, exist_ok=True)
ext = test_set_entries[0]["target"].split('.')[-1]
mext = test_set_entries[0]["mask"].split('.')[-1]

for i, ele in enumerate(test_set_entries):
    target_source = ele['target']
    reference_source = ele['reference']
    mask_source = ele['mask']
    
    target_target = os.path.join(dest_dir, f"pair{i:0>4}_t.{ext}")
    reference_target = os.path.join(dest_dir, f"pair{i:0>4}_r.{ext}")
    mask_target = os.path.join(dest_dir, f"pair{i:0>4}_m.{mext}")
    
    shutil.copy(target_source, target_target)
    shutil.copy(reference_source, reference_target)
