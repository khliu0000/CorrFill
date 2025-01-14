# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List
import torch
import math
import torch.nn.functional as F

from . import share
import diffusers

# CF_IPA_Inpaint
from ip_adapter import IPAdapterPlus
from ip_adapter.attention_processor import AttnProcessor2_0 as IpaProcessor2_0
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .pbe_pipeline import PaintByExamplePipeline


class CorrFill_Inpaint:

    def __init__(self, sd_pipe, device):

        self.device = device
        if isinstance(sd_pipe, PaintByExamplePipeline):
            sd_pipe.vae = sd_pipe.vae.to(self.device)
            sd_pipe.unet = sd_pipe.unet.to(self.device)
            self.pipe = sd_pipe
        else:
            self.pipe = sd_pipe.to(self.device)

        global blk_num
        blk_num = 0
        corr_replace_verbose = False
        for name, block in self.pipe.unet.named_modules():
            if isinstance(block, (diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D, diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D, diffusers.models.unet_2d_blocks.UNetMidBlock2DCrossAttn)):
                for attn_name, attn in block.named_modules():
                    if 'attn1' not in attn_name:
                        continue
                    if isinstance(attn, diffusers.models.attention_processor.Attention):
                        blk_num += 1
                        if isinstance(attn.processor, diffusers.models.attention_processor.AttnProcessor2_0):
                            if corr_replace_verbose:
                                print(f"replace {attn.processor} with ", sep='')
                            attn.processor = CorrGuidanceAttnProcessor2_0()
                            if corr_replace_verbose:
                                print(f"{type(attn.processor)}")
                        elif isinstance(attn.processor, diffusers.models.attention_processor.AttnProcessor):
                            if corr_replace_verbose:
                                print(f"replace {attn.processor} with ", sep='')
                            attn.processor = CorrGuidanceAttnProcessor()
                            if corr_replace_verbose:
                                print(f"{type(attn.processor)}")
                            raise NotImplementedError("Self-guidance is not implemented for this attention processor")
                        else:
                            raise NotImplementedError("Self-guidance is not implemented for this attention processor")
        share.blk_num = blk_num

    def generate(
        self,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=0,
        guidance_scale=7.5,
        num_inference_steps=30,
        my_args=None,
        pbe=False,
        **kwargs,
    ):
        num_prompts = 1

        prompt = ""  # empty prompt
        # Other choices
        # prompt = "best quality, high quality"  # default_p
        # prompt = "Inpaint the given image with visually coherent and high-fidelity background and texture"  # inpaint_p
        # prompt = "The whole image is splited into two parts with the same size, they share the same scene/landmark captured with different viewpoints and times"  # ref_p
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        share.iter_num = num_inference_steps
        share.out_dir = my_args['out_dir']
        share.pair_num = my_args['idx']

        global corr_path, img_height, blk_cnt, blk_num, iter_cnt, iter_num
        corr_path = my_args['corr_gt']
        blk_cnt = 0
        blk_num = share.blk_num
        iter_cnt = 0
        iter_num = share.iter_num
        img_height = kwargs['height']

        # LeftRefill
        if share.leftrefill:
            if share.leftrefill_prompt is None:
                share.leftrefill_prompt = {'prompt_embeds': torch.load("./leftrefill/leftrefill_pos_prompt.pt"),
                                           'negative_prompt_embeds': torch.load("./leftrefill/leftrefill_neg_prompt.pt")}
            images = self.pipe(
                prompt_embeds=share.leftrefill_prompt['prompt_embeds'],
                negative_prompt_embeds=share.leftrefill_prompt['negative_prompt_embeds'],
                guidance_scale=2.5,
                num_inference_steps=num_inference_steps,
                do_corr_guidance=share.latent_optimize,
                cg_t_start=0,
                cg_t_end=share.stop_step_o,
                **kwargs,
            ).images
        else:
            if pbe:
                images = self.pipe(
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.manual_seed(seed),
                    do_corr_guidance=share.latent_optimize,
                    cg_t_start=0,
                    cg_t_end=share.stop_step_o,
                    **kwargs,
                ).images
            else:  # default
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=3,
                    num_inference_steps=num_inference_steps,
                    do_corr_guidance=share.latent_optimize,
                    cg_t_start=0,
                    cg_t_end=share.stop_step_o,
                    **kwargs,
                ).images

        return images


class CF_IPA_Inpaint(IPAdapterPlus):
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to('cpu', dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()

        global blk_num
        blk_num = 0
        corr_replace_verbose = False
        for name, block in self.pipe.unet.named_modules():
            if isinstance(block, (diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D, diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D, diffusers.models.unet_2d_blocks.UNetMidBlock2DCrossAttn)):
                for attn_name, attn in block.named_modules():
                    if 'attn1' not in attn_name:
                        continue
                    if isinstance(attn, diffusers.models.attention_processor.Attention):
                        blk_num += 1
                        if isinstance(attn.processor, diffusers.models.attention_processor.AttnProcessor2_0):  # support ip-adapter
                            if corr_replace_verbose:
                                print(f"replace {attn.processor} with ", sep='')
                            attn.processor = CorrGuidanceAttnProcessor2_0()
                            if corr_replace_verbose:
                                print(f"{type(attn.processor)}")
                        elif isinstance(attn.processor, IpaProcessor2_0):  # support ip-adapter
                            if corr_replace_verbose:
                                print(f"replace {attn.processor} with ", sep='')
                            attn.processor = IpaCgAttnProcessor2_0()
                            if corr_replace_verbose:
                                print(f"{type(attn.processor)}")
                        elif isinstance(attn.processor, diffusers.models.attention_processor.AttnProcessor):
                            if corr_replace_verbose:
                                print(f"replace {attn.processor} with ", sep='')
                            attn.processor = CorrGuidanceAttnProcessor()
                            if corr_replace_verbose:
                                print(f"{type(attn.processor)}")
                            raise NotImplementedError("Self-guidance is not implemented for this attention processor")
                        else:
                            raise NotImplementedError("Self-guidance is not implemented for this attention processor")
        share.blk_num = blk_num

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=0,
        guidance_scale=7.5,
        num_inference_steps=30,
        my_args=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        share.iter_num = num_inference_steps
        share.out_dir = my_args['out_dir']
        share.pair_num = my_args['idx']

        global corr_path, img_height, blk_cnt, blk_num, iter_cnt, iter_num
        corr_path = my_args['corr_gt']
        blk_cnt = 0
        blk_num = share.blk_num
        iter_cnt = 0
        iter_num = share.iter_num
        img_height = kwargs['height']

        torch.cuda.empty_cache()
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            do_corr_guidance=share.latent_optimize,
            cg_t_start=0,
            cg_t_end=share.stop_step_o,
            **kwargs,
        ).images

        return images


corr_path = None
blk_cnt = 0
blk_num = 0
iter_cnt = 0
iter_num = 0
img_height = 512

keep_corr = None
keep_dict = dict()
keep_bank = None
keep_bank_stat = dict()
keep_suppress = None


# Not used in reproduction
class CorrGuidanceAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: diffusers.models.attention_processor.Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IpaCgAttnProcessor2_0(IpaProcessor2_0):
    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale: float = 1.0,
            *args,
            **kwargs,
            ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        # Original AttnProcessor2.0
        if (not share.latent_optimize and not share.attn_masking):
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states, scale=scale)
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        ### CorrFill AttnProcessor
        query_ = attn.head_to_batch_dim(query)
        key_ = attn.head_to_batch_dim(key)
        scores_ = attn.get_attention_scores(query_, key_, attention_mask)  # after softmax
        scores_ = scores_.reshape(len(query), -1, *scores_.shape[1:]).mean(1)
        h = math.isqrt(scores_.shape[1]//2)
        w = h*2
        scores_ = scores_.reshape(len(scores_), h, w, h, w)
        scores_ = scores_[1, :, h:, :, :h]

        global blk_cnt, blk_num, iter_cnt, iter_num
        new_input = (blk_cnt==0) and (iter_cnt==0)
        if new_input:
            share.corr_height = h
        corr_h = share.corr_height

        # Downscale
        if corr_h != h:
            score_down_ratio = (corr_h//h)
            scores_ = F.interpolate(scores_.reshape(1, h*h, h, h),
                                    size=[corr_h, corr_h], mode="bilinear").reshape(1, h, h, -1)
            scores_ /= (score_down_ratio**2)
            scores_ = F.interpolate(scores_.permute(0, 3, 1, 2), size=[corr_h, corr_h],
                                    mode="nearest-exact").permute(0, 2, 3, 1).squeeze().reshape(corr_h, corr_h, corr_h, corr_h)

        if blk_cnt in share.collect_blk:
            try:
                del attn._aux['attn']
            except Exception:
                pass
            attn._aux = {'attn': scores_}
            if blk_cnt == share.collect_blk[-1]:
                query = query.detach()
                key = key.detach()
                value = value.detach()
                residual = residual.detach()
                hidden_states = hidden_states.detach()

        # Attn masking
        global keep_corr, keep_dict, keep_bank, keep_bank_stat, keep_suppress
        skip_mani = get_keep_corr_softmax(scores_, corr_h, h, new_input)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if share.forward_attn_masking and keep_corr is not None and not skip_mani and iter_cnt<share.stop_step_m:
            attn_mask = get_attn_mask(keep_corr, keep_suppress, h, w, corr_h, query.device, query.dtype, boost=share.boost, depress=share.depress)
            if attention_mask is not None:
                attn_mask = attention_mask + attn_mask
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        blk_cnt += 1
        if share.forward_attn_masking and (blk_cnt==blk_num):
            iter_cnt += 1
        blk_cnt %= blk_num
        ### END CorrFill

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CorrGuidanceAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: diffusers.models.attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale: float = 1.0,
            ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        # Original AttnProcessor2.0
        if (not share.latent_optimize and not share.attn_masking):
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states, scale=scale)
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        ### CorrFill AttnProcessor
        query_ = attn.head_to_batch_dim(query)
        key_ = attn.head_to_batch_dim(key)
        scores_ = attn.get_attention_scores(query_, key_, attention_mask)  # after softmax
        scores_ = scores_.reshape(len(query), -1, *scores_.shape[1:]).mean(1)
        h = math.isqrt(scores_.shape[1]//2)
        w = h*2
        scores_ = scores_.reshape(len(scores_), h, w, h, w)
        scores_ = scores_[1, :, h:, :, :h]

        global blk_cnt, blk_num, iter_cnt, iter_num
        new_input = (blk_cnt==0) and (iter_cnt==0)
        if new_input:
            share.corr_height = h
        corr_h = share.corr_height

        # Downscale
        if corr_h != h:
            score_down_ratio = (corr_h//h)
            scores_ = F.interpolate(scores_.reshape(1, h*h, h, h),
                                    size=[corr_h, corr_h], mode="bilinear").reshape(1, h, h, -1)
            scores_ /= (score_down_ratio**2)
            scores_ = F.interpolate(scores_.permute(0, 3, 1, 2), size=[corr_h, corr_h],
                                    mode="nearest-exact").permute(0, 2, 3, 1).squeeze().reshape(corr_h, corr_h, corr_h, corr_h)

        if blk_cnt in share.collect_blk:
            try:
                del attn._aux['attn']
            except Exception:
                pass
            attn._aux = {'attn': scores_}
            if blk_cnt == share.collect_blk[-1]:
                query = query.detach()
                key = key.detach()
                value = value.detach()
                residual = residual.detach()
                hidden_states = hidden_states.detach()

        ### Attn masking
        global keep_corr, keep_dict, keep_bank, keep_bank_stat, keep_suppress
        skip_mani = get_keep_corr_softmax(scores_, corr_h, h, new_input)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if share.forward_attn_masking and keep_corr is not None and not skip_mani and iter_cnt<share.stop_step_m:
            attn_mask = get_attn_mask(keep_corr, keep_suppress, h, w, corr_h, query.device, query.dtype, boost=share.boost, depress=share.depress)
            if attention_mask is not None:
                attn_mask = attention_mask + attn_mask
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        blk_cnt += 1
        if share.forward_attn_masking and (blk_cnt==blk_num):
            iter_cnt += 1
        blk_cnt %= blk_num
        ### END CorrFill

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@torch.no_grad()
def get_attn_mask(corr_in,
                  suppress_in,
                  h,
                  w,
                  corr_h,
                  device,
                  dtype,
                  boost=0,
                  depress=-torch.inf,
                  dis_threshold=0.15,
                  dis_sup_thres=0.05,
                  ):
    corr = corr_in.clone()
    suppress = suppress_in.clone() if suppress_in is not None else None
    if corr_h != h:
        corr = nan_downscale(corr, h, corr_h)
        if suppress is not None:
            suppress = nan_downscale(suppress, h, corr_h)
    space = torch.linspace(0, 1, steps=(h+1))[:-1]
    dynamic_threshold = space[1]
    space += space[1]/2  # Place space at center
    mgrid = torch.meshgrid(space, space, indexing='ij')
    grid = torch.stack(mgrid, dim=-1).to(device=device)
    dis_threshold = dynamic_threshold*share.attn_mask_window if share.attn_mask_window>0 else (max(dynamic_threshold/2, -share.attn_mask_window))
    dis_sup_thres = dynamic_threshold*share.attn_mask_window if share.attn_mask_window>0 else (max(dynamic_threshold/2, -share.attn_mask_window))
    distances = (grid[:, :, None, ...] - corr.reshape(h**2, 2)[None, ...]).norm(dim=-1).reshape(h, h, h, h).permute(2, 3, 0, 1)
    if suppress is not None:
        distances_sup = (grid[:, :, None, ...] - suppress.reshape(h**2, 2)[None, ...]).norm(dim=-1).reshape(h, h, h, h).permute(2, 3, 0, 1)
    # To ensure no nan attends in calculation, can be deleted after experiment
    assert (corr.isnan().logical_not().all(dim=-1).nonzero()==distances.isnan().logical_not().reshape(h, h, -1).all(dim=-1).nonzero()).all()
    attn_mask = torch.zeros((2, 1, h, w, h, w), device=device, dtype=dtype)
    distances = -1 * (distances-dis_threshold)
    distances[distances>=0] = boost * share.str_masking * share.adaptive_weight  # Boost
    distances[distances<0] = depress * share.str_masking * share.adaptive_weight  # Depress
    if suppress is not None:
        distances[(distances_sup<dis_sup_thres).logical_and(distances_sup.isnan().logical_not()).logical_and(distances.isnan())] = depress * share.str_masking * share.adaptive_weight
    distances[distances.isnan()] = 0
    attn_mask[1, :, :, h:, :, :h] = distances
    attn_mask = attn_mask.reshape(2, 1, h*w, h*w)
    return attn_mask


def corr_coord_2_heatmap(coord, threshold=0.1):
    height = coord.shape[0]
    space = torch.linspace(0, 1, steps=(height+1))[:-1]
    space += space[1]/2  # Place space at center
    mgrid = torch.meshgrid(space, space, indexing='ij')
    grid = torch.stack(mgrid, dim=-1).to(device=coord.device)
    distances = (grid[:, :, None, ...] - coord.reshape(height*height, 2)[None, ...]).norm(dim=-1).reshape(height, height, height, height).permute(2, 3, 0, 1)
    return torch.where(distances<threshold, 1, 0).to(dtype=torch.float16)


def weight_smoothing(corr_h, radius=0.15):
    global keep_bank, keep_bank_stat, keep_corr, blk_cnt, iter_cnt
    conf_map = (keep_bank.reshape(corr_h, corr_h, -1).max(-1)[0])/keep_bank_stat['acc_max']
    space = torch.linspace(0, 1, steps=(corr_h+1))[:-1]
    space += space[1]/2  # Place space at center
    mgrid = torch.meshgrid(space, space, indexing='ij')
    grid = torch.stack(mgrid, dim=-1).to(device=keep_corr.device, dtype=keep_corr.dtype)
    nan_mask = keep_corr.isnan().logical_not().all(dim=-1)  # masked off nan
    real_location = grid[nan_mask]  # norm'ed starting coord of non-nan shape(n,2)
    real_len = nan_mask.sum().item()
    dist_between_real_loc = (real_location[None]-real_location[:, None]).norm(dim=-1)
    window_neighbor = (dist_between_real_loc<radius)  # bool of shape(n,n)
    flow_weight = conf_map[nan_mask]  # shape(n)
    flow_weight_2d = flow_weight.repeat(real_len, 1)
    flow_weight_2d[window_neighbor.logical_not()] = 0
    offset_grid = (keep_corr-grid)[nan_mask]   # norm'ed flow of shape(n,2)
    offset_grid_i = offset_grid[:, 0]
    offset_grid_j = offset_grid[:, 1]
    window_neighbor_weight = flow_weight_2d.sum(-1)
    new_offset_grid_i = (offset_grid_i.repeat(real_len, 1)*flow_weight_2d).sum(-1)/window_neighbor_weight
    new_offset_grid_j = (offset_grid_j.repeat(real_len, 1)*flow_weight_2d).sum(-1)/window_neighbor_weight
    new_corr = keep_corr.clone()
    new_corr[nan_mask] = grid[nan_mask] + torch.stack((new_offset_grid_i, new_offset_grid_j), -1)
    new_corr[((new_corr<0).any(-1)).logical_or((new_corr>=1).any(-1))] = torch.nan
    # For visualization
    if False and ((iter_cnt in [0, 49, share.stop_step_m]) and (blk_cnt==share.vote_blk[-1])) or (iter_cnt==0 and blk_cnt==share.vote_blk[0]):
        temp_corr1 = grid[new_corr.isnan().logical_not().all(-1)]
        temp_corr1 = (temp_corr1*corr_h-0.5).round().to(dtype=int)
        temp_corr1 = temp_corr1[:, 0]*corr_h + temp_corr1[:, 1]
        temp_corr2 = new_corr[new_corr.isnan().logical_not().all(-1)]
        temp_corr2 = (temp_corr2*corr_h-0.5).round().to(dtype=int)
        temp_corr2 = temp_corr2[:, 0]*corr_h + temp_corr2[:, 1]
        return new_corr, temp_corr1, temp_corr2
    return new_corr, None, None


def nan_downscale(in_tensor, to_shape, corr_h):
    assert in_tensor.shape == (corr_h, corr_h, 2)
    assert (corr_h/to_shape)%2 == 0
    down_scale = corr_h//to_shape
    out_tensor = in_tensor.permute(2, 0, 1).view(2, to_shape, down_scale, to_shape, down_scale).permute(0, 1, 3, 2, 4).reshape(2, to_shape, to_shape, -1)
    return out_tensor.nanmean(dim=-1).permute(1, 2, 0)


def get_attention(
    attn: diffusers.models.attention_processor.Attention,
    query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
) -> torch.Tensor:

    dtype = query.dtype
    if attn.upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=attn.scale,
    )
    del baddbmm_input

    if attn.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs


@torch.no_grad()
def get_keep_corr_softmax(scores_, corr_h, h, new_input):
    global blk_cnt, blk_num, iter_cnt, iter_num, real_iter_cnt
    global keep_corr, keep_dict, keep_bank, keep_bank_stat, keep_suppress
    min_number_corr_to_mani = 4
    skip_mani = False
    if new_input:
        keep_corr = None
        keep_dict = dict()
        keep_bank = None
        keep_bank_stat = {'cnt': 0, 'acc_max': 0}
        keep_suppress = None
        share.corr = None
        share.mask2d = None
    if (blk_cnt in share.vote_blk):
        with torch.no_grad():
            first_vote_blk = (blk_cnt==share.vote_blk[0] and iter_cnt==0)

            if first_vote_blk:
                keep_bank = torch.zeros([corr_h, corr_h, corr_h, corr_h], device=scores_.device, dtype=scores_.dtype)

            keep_bank += torch.clamp(scores_-scores_.reshape(corr_h, corr_h, -1).median(-1)[0][..., None, None], min=0.0)  # almost no effect, but elim small number
            keep_bank_stat['cnt'] += 1
            keep_bank_stat['acc_max'] += torch.max(scores_-scores_.reshape(corr_h, corr_h, -1).median(-1)[0][..., None, None]).item()

            keep_corr = torch.full((corr_h, corr_h, 2), torch.nan, device=scores_.device, dtype=scores_.dtype)
            try:
                top_val, top_idx = torch.max(keep_bank.reshape(corr_h**2, corr_h**2), dim=-1)
                nonzero_bool = (top_val!=0.0)
                if (top_val!=0.0).sum().item() != 4096:
                    print("top_val has zeros")
                corr1 = nonzero_bool.nonzero().squeeze()
                corr2 = top_idx[nonzero_bool]
                corr1_spa = torch.stack((corr1//corr_h, corr1%corr_h))
                corr2_spa = torch.stack((corr2//corr_h, corr2%corr_h))
                keep_corr[corr1_spa[0, :], corr1_spa[1, :]] = (corr2_spa.permute(1, 0)/corr_h + 1/corr_h/2).to(dtype=scores_.dtype)  # Place at center

                keep_suppress = keep_corr.clone() if (share.use_df) else None
                if keep_suppress is not None:
                    keep_dict = dict()

                corr_ref_idx = torch.argmax(keep_bank[(keep_corr[keep_corr.isnan().logical_not().all(-1)].reshape(-1, 2)[..., 0]*corr_h-0.5).to(dtype=int).tolist(),
                                                      (keep_corr[keep_corr.isnan().logical_not().all(-1)].reshape(-1, 2)[..., 1]*corr_h-0.5).to(dtype=int).tolist()].reshape(-1, corr_h*corr_h), -1)
                corr_ref_idx = torch.stack([corr_ref_idx//corr_h, corr_ref_idx%corr_h], 1)
                corr_ref = keep_corr.clone()
                corr_ref[keep_corr.isnan().logical_not().all(-1)] = ((corr_ref_idx+0.5)/corr_h).to(dtype=corr_ref.dtype)
                if share.use_df:  # dominant filter
                    dominator_thres = share.dominator_thres
                    df_mask = (((keep_corr[:, :, None, None]-keep_corr[None, None, :, :])==0).all(-1).reshape(corr_h, corr_h, -1).sum(-1) > dominator_thres)
                    keep_bank[(keep_corr[df_mask][..., 0]*corr_h-0.5).to(dtype=int).tolist(), (keep_corr[df_mask][..., 1]*corr_h-0.5).to(dtype=int).tolist(),
                              (corr_ref[df_mask][..., 0]*corr_h-0.5).to(dtype=int).tolist(), (corr_ref[df_mask][..., 1]*corr_h-0.5).to(dtype=int).tolist()] = 0
                    keep_corr[df_mask] = torch.nan
                    if "outliers_mask" not in keep_dict.keys():
                        keep_dict["outliers_mask"] = df_mask
                    else:
                        keep_dict["outliers_mask"] = keep_dict["outliers_mask"].logical_or(df_mask)
                if share.use_ws:  # smoothing
                    new_corr, smooth_corr1, smooth_corr2 = weight_smoothing(corr_h, share.smooth_window)
                    dist_new_old = (keep_corr-new_corr).norm(dim=-1)
                    outlie_thres = share.outlie_thres
                    outlie_mask = (dist_new_old > outlie_thres)
                    keep_corr[outlie_mask] = torch.nan
                    if "outliers_mask" not in keep_dict.keys():
                        keep_dict["outliers_mask"] = outlie_mask
                    else:
                        keep_dict["outliers_mask"] = keep_dict["outliers_mask"].logical_or(outlie_mask)
                    keep_corr = new_corr  # Swap keep_corr (suppress has copy)
                if keep_suppress is not None:
                    keep_suppress[keep_dict["outliers_mask"].logical_not()] = torch.nan
            except Exception:
                # If correspondence voting cannot get enough number of correspondence, then skip manipulation for this attention layer forward pass
                skip_mani = True
                print("guidance skipped")

            skip_mani = False

            if not skip_mani and keep_corr.isnan().all(-1).logical_not().sum()<min_number_corr_to_mani:
                skip_mani = True

            share.mask2d = keep_corr.isnan().all(-1).logical_not()
            if share.mask2d.any():
                share.corr = corr_coord_2_heatmap(keep_corr, 0.05)
                share.corr[share.mask2d.logical_not()] = 0.0

    return skip_mani
