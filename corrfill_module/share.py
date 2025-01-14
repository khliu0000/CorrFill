# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Shared objects
corr = None
mask2d = None
adaptive_weight = 1.0
forward_attn_masking = False

corr_height = -1
collect_blk = []

blk_num = 0
iter_num = 0
leftrefill_prompt = None

leftrefill = False
attn_masking = False
latent_optimize = False
use_adaptive = True
vote_blk = [10, 11, 12, 13, 14]
use_df = False
use_ws = False
stop_step_m = -1
stop_step_o = -1
str_masking = 1.0
str_optimize = 1.0
attn_mask_window = -1
boost = 0
depress = 0
smooth_window = -1
outlie_thres = 0.15
dominator_thres = 4
optimize_splits = 2

out_dir = None
pair_num = -1
r_path = None
t_path = None
m_path = None
embed_path = None
