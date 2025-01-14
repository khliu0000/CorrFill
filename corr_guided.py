# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from diffusers import DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np

import glob
import os
import datetime
from tqdm import tqdm
import argparse
import json

from corrfill_module.corrfill import CorrFill_Inpaint, CF_IPA_Inpaint
# LeftRefill, SideBySide, IPAdapterPlus
from corrfill_module.cg_pipeline import CorrGuidanceInpaintingPipeline
# PaintByExample
from corrfill_module.pbe_pipeline import PaintByExamplePipeline
from corrfill_module import share

# Evaluation
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity
from torchmetrics.functional import peak_signal_noise_ratio
import lpips
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description="Faithfulness enhanced inpainting")

parser.add_argument("--config", metavar="config_file",
                    required=True, dest="config_path",
                    action="store",
                    help="path to config file for validation")
parser.add_argument("--data", metavar="validation_root",
                    required=False, dest="val_root",
                    action="store", default="./corrfill_test/",
                    help="directory of the root of validation sets")
parser.add_argument("--out", metavar="out_path",
                    required=False, dest="out_path",
                    action="store", default="./results/",
                    help="output directory of the inpainting results")

# For IPA
parser.add_argument("--ipa_root", metavar="ipa_root",
                    required=False, dest="ipa_root",
                    action="store", default=".",
                    help="parent directory of IP-Adapter's repo")
parser.add_argument("--sd_root", metavar="sd_root",
                    required=False, dest="sd_root",
                    action="store", default="./runwayml_inpainting/",
                    help="root of the runwayml sd inpainting (for IP-Adapter)")

args = parser.parse_args()


with open(args.config_path) as config_opt_json:
    config_opt = json.load(config_opt_json)

baseline = config_opt["baseline"]

if baseline=="sidebyside" or baseline=="leftrefill":
    base_model_path = "stabilityai/stable-diffusion-2-inpainting"
    noise_scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").to(dtype=torch.float16)
elif baseline == "ipadapterplus":
    ipa_root = args.ipa_root
    sd_root = args.sd_root
    base_model_path = os.path.join(sd_root, "runwayml/stable-diffusion-inpainting")
    image_encoder_path = os.path.join(ipa_root, "IP-Adapter/models/image_encoder")
    ip_ckpt = os.path.join(ipa_root, "IP-Adapter/models/ip-adapter-plus_sd15.bin")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").to(dtype=torch.float16)

device = "cuda"


torch.cuda.empty_cache()
if baseline == "paintbyexample":
    pipe = PaintByExamplePipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
else:
    pipe = CorrGuidanceInpaintingPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )


if __name__ == "__main__":
    path = None
    calculate_metrics = True
    target_pair = None  # Can be used to specific pair to test. E.g. target_pair = {'realestate':[1,3]}
    target_list = None
    total_val_pair = 500  # can be set to smaller values to validate on smaller subsets

    root_path = args.out_path
    path = os.path.join(args.val_root, config_opt["val_set"]+"/")

    filenames = []
    num_dataset = len(path)
    all_pairs = sorted(glob.glob(path+"*"))
    if target_pair is None:
        target = [os.path.join(x, "target.png") for x in all_pairs][:total_val_pair]
        reference = [os.path.join(x, "source.png") for x in all_pairs][:total_val_pair]
        mask = [os.path.join(x, "mask.png") for x in all_pairs][:total_val_pair]
    else:
        target = [os.path.join(x, "target.png") for x in all_pairs]
        reference = [os.path.join(x, "source.png") for x in all_pairs]
        mask = [os.path.join(x, "mask.png") for x in all_pairs]

    assert (len(target) == len(reference)) and (len(target) == len(mask))
    filenames = [target, reference, mask]

    if calculate_metrics:
        loss_fn_alex = lpips.LPIPS(net='alex')
        transform = transforms.Compose([transforms.PILToTensor()])

    exp_name = baseline + "_" + config_opt["val_set"]
    description = ""
    time_str = datetime.datetime.today().strftime("%m%d%H%M%S")+description
    exp_root_path = os.path.join(os.path.join(root_path, exp_name), time_str)
    print(f"Experiment {exp_name} at {time_str}")
    os.makedirs(exp_root_path, exist_ok=True)
    record_file_path = os.path.join(exp_root_path, "exp_record.txt")

    len_dataset = 0
    if target_pair is not None and config_opt["val_set"] in target_pair.keys():
        len_dataset += len(target_pair[config_opt["val_set"]])
    else:
        len_dataset += len(filenames[0])

    if baseline == "ipadapterplus":
        model = CF_IPA_Inpaint(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    else:
        model = CorrFill_Inpaint(pipe, device=device)

    with tqdm(total=len_dataset) as pbar:
        dataset_name = config_opt["val_set"]
        out_dir = os.path.join(exp_root_path, dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        target_list = None
        if (target_pair is not None) and (dataset_name in target_pair.keys()):
            target_list = target_pair[dataset_name]
            if not isinstance(filenames, np.ndarray):
                filenames = np.array(filenames)[:, target_list]

        with open(record_file_path, 'a') as record_file:
            record_file.write(f"Dataset {dataset_name}\n")

        psnr = []
        lpipss = []
        ssim = []
        with open(record_file_path, 'a') as record_file:
            share.use_df = config_opt["dominant_filter"]
            share.use_ws = config_opt["smoothing"]
            share.attn_masking = config_opt["attn_masking"]
            share.latent_optimize = config_opt["latent_optimize"]
            share.str_masking = config_opt["str_masking"]
            share.str_optimize = config_opt["str_optimize"]
            random_seed = config_opt["random_seed"]
            share.stop_step_m = config_opt["step_masking"]
            share.stop_step_o = config_opt["step_optimize"]
            share.attn_mask_window = config_opt["mask_window"]
            share.boost = config_opt["mask_boost"]
            share.smooth_window = config_opt["smooth_window"]
            share.vote_blk = list(range(config_opt["vote_blocks"][0], config_opt["vote_blocks"][1]+1))
            share.leftrefill = True if baseline=="leftrefill" else False
            share.depress = -torch.inf
            record_file.write(f"Filter df:{share.use_df}/smoothing:{share.use_ws} " +
                              f"Toggle attn_masking:{share.attn_masking}/latent_optimize:{share.latent_optimize}  " +
                              f"Boost:{share.boost}  SmoothWindow:{share.smooth_window}  Step:{share.stop_step_m}/{share.stop_step_o}  " +
                              f"Seed:{random_seed}   MaskWindow:{share.attn_mask_window}  " +
                              f"Strength attn_masking:{share.str_masking}/latent_optimize:{share.str_optimize}  " +
                              f"Baseline:{baseline} VoteBlk:{share.vote_blk}\n")

            if baseline=="leftrefill" or baseline=="sidebyside":
                share.optimize_splits = 1
            else:
                share.optimize_splits = 2
                if baseline == "paintbyexample":
                    guidance_scale = 5

        outter_j = 0
        for t_path, r_path, m_path in zip(filenames[0], filenames[1], filenames[2]):
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            corr_gt_dir = None
            if target_list is not None:
                j = target_list[outter_j]
            else:
                j = outter_j
            my_args = {'out_dir': out_dir, 'idx': j, 'corr_gt': corr_gt_dir}

            if baseline == "paintbyexample":
                embed_root = f"./pbe_embed/{dataset_name}/"
                os.makedirs(embed_root, exist_ok=True)
                share.embed_path = os.path.join(embed_root, f"{j:0>4}")
            elif baseline == "ipadapterplus":
                embed_root = f"./ipa_embed/{dataset_name}/"
                os.makedirs(embed_root, exist_ok=True)
                share.embed_path = os.path.join(embed_root, f"{j:0>4}")

            share.r_path, share.t_path, share.m_path = r_path, t_path, m_path
            reference = Image.open(r_path)
            target = Image.open(t_path)
            mask = Image.open(m_path)
            mask = np.array(mask)
            mask[mask>0] = 255
            mask = Image.fromarray(np.uint8(mask))
            masked_t = Image.composite(mask, target, mask.convert('L'))
            left = reference
            right = masked_t
            right_m = mask.convert('L')

            image = Image.new('RGB', (2*left.size[0], left.size[1]))
            image.paste(left, (0, 0))
            image.paste(right, (left.size[0], 0))
            wide_mask = Image.new('L', (2*right_m.size[0], right_m.size[1]))
            wide_mask.paste(right_m, (right_m.size[0], 0))
            box = (left.size[0], 0, 2*left.size[0], left.size[1])

            auto_retry_before_increase_split = 3
            original_splits = share.optimize_splits
            retries = 0
            success = 0
            while True:  # OOM handler
                try:
                    if baseline=="sidebyside" or baseline=="leftrefill":
                        generated = model.generate(eta=1, num_samples=1, num_inference_steps=50, image=image, mask_image=wide_mask, strength=1, height=left.size[1], width=2*left.size[0], my_args=my_args)
                    elif baseline == "paintbyexample":
                        if share.latent_optimize or share.attn_masking:
                            generated = model.generate(guidance_scale=guidance_scale, num_inference_steps=50, image=image, mask_image=wide_mask, example_image=reference, pbe=True, height=left.size[1], width=2*left.size[0], my_args=my_args, seed=random_seed)
                        else:
                            generated = model.generate(guidance_scale=5, num_inference_steps=50, image=target, mask_image=mask, example_image=reference, pbe=True, height=left.size[1], width=left.size[0], my_args=my_args, seed=random_seed)
                    else:
                        if share.latent_optimize or share.attn_masking:
                            generated = model.generate(pil_image=reference, eta=1, num_samples=1, num_inference_steps=50, image=image, mask_image=wide_mask, strength=1, height=left.size[1], width=2*left.size[0], my_args=my_args)
                        else:
                            generated = model.generate(pil_image=reference, eta=1, num_samples=1, num_inference_steps=50, image=image.crop(box), mask_image=mask, height=left.size[1], width=left.size[0], my_args=my_args)
                    success += 1
                    if success == auto_retry_before_increase_split:
                        share.optimize_splits = original_splits
                    break
                except RuntimeError as e:
                    from time import sleep
                    if 'out of memory' in str(e):
                        print(f"encounter OOM at pari {j}, auto retrying")
                        sleep(1)
                        retries += 1
                        success = 0
                        if retries == auto_retry_before_increase_split:
                            share.optimize_splits = min(2*share.optimize_splits, share.blk_num)
                            retries = 0
                    else:
                        print(f"Catched error {str(e)}")
                        raise e

            if baseline=="leftrefill" or baseline=="sidebyside" or share.latent_optimize or share.attn_masking:
                generated[0].save(os.path.join(out_dir, f"pair{j:0>4}_whole.png"))
                cropped = generated[0].crop(box)
                cropped.save(os.path.join(out_dir, f"pair{j:0>4}_out.png"))
                merge_target = Image.composite(cropped, target, mask.convert('L'))
                merge_target.save(os.path.join(out_dir, f"pair{j:0>4}_target.png"))
                if True:
                    target.save(os.path.join(out_dir, f"pair{j:0>4}_gt.png"))
                    image.save(os.path.join(out_dir, f"pair{j:0>4}_masked.png"))
                if calculate_metrics:
                    pred = transform(merge_target).to(dtype=torch.float32)/255
                    gt = transform(target).to(dtype=torch.float32)/255

                    psnr_ = peak_signal_noise_ratio(pred, gt, data_range=1.0)
                    lpips_ = loss_fn_alex(pred*2-1, gt*2-1)  # lpips needs -1~1
                    pred_np = TF.rgb_to_grayscale(pred)[0].cpu().numpy()
                    origin_np = TF.rgb_to_grayscale(gt)[0].cpu().numpy()
                    ssim_ = structural_similarity(pred_np, origin_np, data_range=1.0)
                    psnr.append(psnr_.item())
                    ssim.append(ssim_)
                    lpipss.append(lpips_.item())
            else:
                generated[0].save(os.path.join(out_dir, f"pair{j:0>4}_out.png"))
                merge_target = Image.composite(generated[0], target, mask.convert('L'))
                merge_target.save(os.path.join(out_dir, f"pair{j:0>4}_target.png"))
                if True:
                    target.save(os.path.join(out_dir, f"pair{j:0>4}_gt.png"))
                    image.crop(box).save(os.path.join(out_dir, f"pair{j:0>4}_masked.png"))
                if calculate_metrics:
                    pred = transform(merge_target).to(dtype=torch.float32)/255
                    gt = transform(target).to(dtype=torch.float32)/255

                    psnr_ = peak_signal_noise_ratio(pred, gt, data_range=1.0)
                    lpips_ = loss_fn_alex(pred*2-1, gt*2-1)  # lpips needs -1~1
                    pred_np = TF.rgb_to_grayscale(pred)[0].cpu().numpy()
                    origin_np = TF.rgb_to_grayscale(gt)[0].cpu().numpy()
                    ssim_ = structural_similarity(pred_np, origin_np, data_range=1.0)
                    psnr.append(psnr_.item())
                    ssim.append(ssim_)
                    lpipss.append(lpips_.item())

            if calculate_metrics and outter_j%100 == 0:
                print(f"intermediate {outter_j}\n{np.mean(psnr)}\n{np.mean(ssim)}\n{np.mean(lpipss)}")

            outter_j = outter_j + 1
            pbar.update(1)
            if target_pair is not None and outter_j==max([x for x in target_pair.values()]):
                import pdb; pdb.set_trace()
        used_time = pbar.format_dict['elapsed']
        num_pbar = pbar.format_dict['total']
        if calculate_metrics:
            print('PSNR:', np.mean(psnr))
            print('SSIM:', np.mean(ssim))
            print('LPIPS:', np.mean(lpipss))
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"Evaluation {np.mean(psnr)}/{np.mean(ssim)}/{np.mean(lpipss)}  sample:{total_val_pair} time:{used_time/num_pbar}/{used_time}\n")

print(exp_root_path)
