# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
import torch
import cv2
import numpy as np
import scipy.io as sio
import random
import glob
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler
from src.utils.misc import lower_config

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine

def draw_match(img1_path, img2_path, corr1, corr2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1PIL = Image.open(img1_path)
    maskPIL = Image.new('RGB',(img1PIL.size[0], img1PIL.size[1]))
    retry = 5
    max_box_width = 256
    max_box_height = 256
    box_margin = 64 # 128
    min_num_vertex = 8
    max_num_vertex = 12
    min_vertex_width = 24
    max_vertex_width = 80
    num_vertice_group = 3
    corr_threshold = 80

    if len(corr1)<corr_threshold:
        print(f"***** number of keypoints {len(corr1)} less than nkp {corr_threshold} *****")
        return None, None
    num_vertex = np.random.randint(min_num_vertex, max_num_vertex)  
    width = np.random.randint(min_vertex_width, max_vertex_width)
    idx = random.sample(range(len(corr1)), num_vertex)
    white = (255,255,255)

    box_t = int(np.min(corr1[:,1]))
    box_b = int(np.max(corr1[:,1]))
    box_l = int(np.min(corr1[:,0]))
    box_r = int(np.max(corr1[:,0]))
    if (box_b-box_t) <= max_box_height:
        max_box_height = (box_b-box_t)
    if (box_r-box_l) <= max_box_width:
        max_box_width = (box_r-box_l)
    if max_box_width<=0 or max_box_height<=0:
        import pdb; pdb.set_trace()

    while retry>0:
        box_width = np.random.randint(max(1, max_box_width-box_margin), max_box_width)
        box_height = np.random.randint(max(1, max_box_height-box_margin), max_box_height)
        offset_w = box_l+np.random.randint(0, (box_r-box_l)-box_width)
        offset_h = box_t+np.random.randint(0, (box_b-box_t)-box_height)

        top, bottom, left, right = offset_h, offset_h+box_height, offset_w, offset_w+box_width
        mid_h = int((top+bottom)/2)
        mid_w = int((left+right)/2)
        topside = np.logical_and(corr1[:,1]>=top, corr1[:,1]<=mid_h)
        botside = np.logical_and(corr1[:,1]>=mid_h, corr1[:,1]<=bottom)
        leftside = np.logical_and(corr1[:,0]>=left, corr1[:,0]<=mid_w)
        rightside = np.logical_and(corr1[:,0]>=mid_w, corr1[:,0]<=right)
        centerside = np.logical_and(np.logical_and(corr1[:,1]>=(top+mid_h)/2, corr1[:,1]<=(mid_h+bottom)/2), np.logical_and(corr1[:,0]>=(left+mid_w)/2, corr1[:,0]<=(mid_w+right)/2))
        tlcount = len(corr1[np.logical_and(topside, leftside)])
        trcount = len(corr1[np.logical_and(topside, rightside)])
        blcount = len(corr1[np.logical_and(botside, leftside)])
        brcount = len(corr1[np.logical_and(botside, rightside)])
        ctcount = len(corr1[centerside])
        if tlcount<1 or trcount<1 or blcount<1 or brcount<1 or ctcount<1:
            retry -= 1
            if retry == 0:
                print(f"fail to produce BBox")
                return None, None
        else:
            break

    vertex1 = corr1[idx]
    vertex1 = np.sort(vertex1, axis=0)
    vertice_group = sorted(random.sample(range(len(vertex1))[1:-1], num_vertice_group-1))
    vertice_group = [0]+vertice_group+[len(vertex1)]
    draw = ImageDraw.Draw(maskPIL)
    #import pdb; pdb.set_trace()
    
    for i in range(len(vertice_group)-1):
        vertice = vertex1[vertice_group[i]:vertice_group[i+1]]
        draw.line(vertice, fill=white, width=width)
        for v in vertice:
            draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=white)
    
    #vertex2 = corr2[idx]
    corr1 = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    
    mask = np.array(maskPIL)[...,::-1]
    mask[offset_h:offset_h+box_height, offset_w:offset_w+box_width] = 255
    img1[mask==255] = 255

    display = cv2.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              flags=4
                              )
    
    
    return display, mask

if __name__ == '__main__':
# run QuadTreeAttention setup.py in the other folder first !!!
    parser = argparse.ArgumentParser(
        description='QuadTreeAttention demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weight', type=str, default="outdoor.ckpt", help="Path to the checkpoint.")
    parser.add_argument('--config_path', type=str, default="configs/loftr/outdoor/loftr_ds_quadtree.py", help="Path to the config.")
    parser.add_argument('--data_dir', type=str, default="source/", help="data directory")
    parser.add_argument('--output_dir', type=str, default='./out/', help="out directory")

    opt = parser.parse_args()

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(opt.config_path)
    _config = lower_config(config)

    # Matcher: LoFTR
    matcher = LoFTR(config=_config['loftr'])
    state_dict = torch.load(opt.weight, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict, strict=True)


    random.seed(42)
    np.random.seed(42)
    path = {}
    path['realestate'] = "validation500pairs/realestate/"
    path['megadepth'] = "validation500pairs/megadepth/"
    filenames = {}
    num_dataset = len(path)
    for k, v in path.items():
        img1_list = sorted(glob.glob(os.path.join(v, "pair*_t.*")))
        img2_list = sorted(glob.glob(os.path.join(v, "pair*_r.*")))
        
        assert (len(img1_list) == len(img2_list))
        filenames[k] = [img1_list, img2_list]
        
    save_path = "validation500pairs/"

    for dataset_name, all_path in filenames.items():
        out_pth = f"{save_path}{dataset_name}/"
        os.makedirs(out_pth, exist_ok=True)
        img0_list = all_path[0]
        img1_list = all_path[1]
        total_pair = 0
        
        with tqdm(total=len(img0_list)) as pbar:
            for i in range(len(img0_list)):
                img0_pth = img0_list[i]
                img1_pth = img1_list[i]
                prefix = img0_pth.split("/")[-1].split("_")[0]
                masked_out_pth = os.path.join(out_pth, f"{prefix}_tm.png")
                mask_out_pth = os.path.join(out_pth, f"{prefix}_m.png")
                img0_out_pth = os.path.join(out_pth, f"{prefix}_t.png")
                img1_out_pth = os.path.join(out_pth, f"{prefix}_r.png")
                corr_pth = os.path.join(out_pth, f"{prefix}_corr.png")
                
                img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
                img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

                img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
                batch = {'image0': img0, 'image1': img1}

                # Inference with LoFTR and get prediction
                with torch.no_grad():
                    matcher.eval()
                    matcher.to('cuda')
                    matcher(batch)
                    mkpts0 = batch['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch['mkpts1_f'].cpu().numpy()
                    mconf = batch['mconf'].cpu().numpy()
                    mask_conf = mconf > 0
                    conf = mconf[mask_conf]

                dict_to_save = {}        
                dict_to_save['kpt1'] = mkpts0
                dict_to_save['kpt2'] = mkpts1
                dict_to_save['conf'] = mconf
                threshold = 0.8
                #import pdb; pdb.set_trace()
                
                corr1 = mkpts0[mconf>0.8]
                corr2 = mkpts1[mconf>0.8]
                
                display, mask = draw_match(img0_pth, img1_pth, corr1, corr2)
                if display is not None:
                    masked = cv2.imread(img0_pth)
                    masked[mask==255] = 255
                    cv2.imwrite(corr_pth, display)
                    cv2.imwrite(masked_out_pth, masked)
                    cv2.imwrite(mask_out_pth, mask)
                    total_pair += 1
                pbar.update(1)
            print(total_pair)
        

