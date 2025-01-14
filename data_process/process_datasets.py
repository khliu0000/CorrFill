# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import shutil
import os
import glob
import random
from PIL import Image

path = {}
path['megadepth'] = "megadepth"
path['realestate'] = "realestate"
out_dir = "validation500pairs/LeftRefill_test"
pairs = 500
random.seed(0)

filenames = {}
for k, v in path.items():
    pair_list = sorted(glob.glob(os.path.join(v, "pair*_corr.*")))
    pair_list = sorted(random.sample(pair_list, pairs))
    img1_list_tm = [x.replace("_corr", "_tm") for x in pair_list]
    if k == "megadepth":
        img1_list = [x.replace("_corr.png", "_t.jpg") for x in pair_list]
        img2_list = [x.replace("_corr.png", "_r.jpg") for x in pair_list]
    else:
        img1_list = [x.replace("_corr", "_t") for x in pair_list]
        img2_list = [x.replace("_corr", "_r") for x in pair_list]
    mask_list = [x.replace("_corr", "_m") for x in pair_list]
    assert (len(img1_list) == len(img2_list))
    filenames[k] = [img1_list, img2_list, mask_list, img1_list_tm]

for dataset_name, all_path in filenames.items():
    dest = os.path.join(out_dir, dataset_name)
    os.makedirs(dest, exist_ok=True)
    for t, r, m, tm in zip(all_path[0], all_path[1], all_path[2], all_path[3]):
        pair_num = t.split("/")[-1].split("_")[0]
        pair_num2 = r.split("/")[-1].split("_")[0]
        pair_num3 = m.split("/")[-1].split("_")[0]
        if r[-3:] == "jpg":
            png = Image.open(r)
            r = r.replace("jpg", "png")
            png.save(r)
        assert (pair_num==pair_num2) and (pair_num2==pair_num3)
        dest2 = os.path.join(dest, pair_num)
        os.makedirs(dest2, exist_ok=True)
        shutil.copyfile(t, os.path.join(dest2, "target.png"))
        shutil.copyfile(r, os.path.join(dest2, "source.png"))
        shutil.copyfile(m, os.path.join(dest2, "mask.png"))
        shutil.copyfile(tm, os.path.join(dest2, "masked_target.png"))