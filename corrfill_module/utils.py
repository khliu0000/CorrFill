# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch


def bce_loss(pred_in, gt_in, mask=None, reduction='mean'):
    assert pred_in.size() == gt_in.size()
    if mask is not None:
        pred = pred_in[mask]
        gt = gt_in[mask]
    pos = (gt==1).float()
    neg = (gt==0).float()
    num_pos = pos.sum()
    num_neg = neg.sum()
    alpha_pos = num_neg / (num_pos+num_neg)
    alpha_neg = num_pos / (num_pos+num_neg)
    weights = alpha_pos*pos + alpha_neg*neg

    eps = 1e-7
    pred_max = (pred.flatten(-2, -1)+eps).max(-1)[0].detach()
    pred_mdn = (pred.flatten(-2, -1)+eps).median(-1)[0].detach()
    pred_norm = torch.clamp(((pred-pred_mdn[:, None, None])/pred_max[:, None, None]), min=0.0, max=1.0)

    return torch.nn.functional.binary_cross_entropy_with_logits(pred_norm, gt, weights, reduction=reduction)
