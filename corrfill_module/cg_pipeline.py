# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from diffusers import StableDiffusionInpaintPipeline
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AsymmetricAutoencoderKL
from collections import defaultdict

from .utils import bce_loss
from . import share


class CorrGuidanceInpaintingPipeline(StableDiffusionInpaintPipeline):
    # Template from https://dave.ml/selfguidance/
    def get_cg_aux(self, cfg=True, transpose=True):
        aux = defaultdict(dict)
        for name, aux_module in self.unet.named_modules():
            try:
                module_aux = aux_module._aux
                if transpose:
                    for k, v in module_aux.items():
                        aux[k][name] = v
                else:
                    aux[name] = module_aux
            except AttributeError:
                pass
        return aux

    def wipe_cg_aux(self):
        for name, aux_module in self.unet.named_modules():
            try:
                del aux_module._aux
            except AttributeError:
                pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        cg_t_start=0,
        cg_t_end=0,
        cg_loss_rescale=1000.0,
        cg_grad_wt=1.0,
        do_corr_guidance=True,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image

        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width)

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        ### CorrFill
        self.wipe_cg_aux()
        torch.cuda.empty_cache()
        ### End CorrFill

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # CorrFill
                do_corr_guidance_iter = do_corr_guidance and cg_t_start <= i < cg_t_end

                # First run of UNet to collect gradients for latent optimizing
                if do_corr_guidance_iter:
                    with torch.set_grad_enabled(do_corr_guidance_iter):
                        share.forward_attn_masking = False
                        self.unet.eval()
                        self.unet.requires_grad_(False)

                        grad_store = []
                        splits = share.optimize_splits
                        splited_blk = np.arange(share.blk_num).reshape(splits, -1).tolist()
                        self.unet.enable_gradient_checkpointing()
                        for split_idx in range(splits):
                            torch.cuda.empty_cache()
                            latents_left, latents_right = latents.chunk(2, dim=-1)
                            latents_right.requires_grad_(True)
                            latents = torch.cat([latents_left, latents_right], dim=-1)
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                            # concat latents, mask, masked_image_latents in the channel dimension
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                            if num_channels_unet == 9:
                                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                            share.collect_blk = splited_blk[split_idx]
                            _ = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                            )[0].detach()
                            if share.mask2d.any():
                                cg_aux = self.get_cg_aux(do_classifier_free_guidance)
                                cg_loss = torch.stack([bce_loss(v, share.corr, share.mask2d) for k, v in cg_aux['attn'].items()]).mean()
                                grad = torch.autograd.grad(cg_loss_rescale*cg_loss, latents_right)[0] / cg_loss_rescale
                                assert not grad.isnan().any(), "grad has nan, enable ananomy detection to find the problem"
                                grad_store.append(grad)
                                cg_loss = cg_loss.detach()
                            latent_model_input = latent_model_input.detach()
                            latents = latents.detach()
                            latents_right = latents_right.detach()
                            self.wipe_cg_aux()
                        if len(grad_store) > 0:
                            cg_grad = torch.cat(grad_store).mean(0, keepdim=True)
                        else:
                            cg_grad = None

                share.forward_attn_masking = share.attn_masking
                torch.cuda.empty_cache()
                self.unet.eval()
                self.unet.requires_grad_(False)
                ### End CorrFill

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # CorrFill
                self.unet.disable_gradient_checkpointing()
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                ### CorrFill
                if share.attn_masking or share.latent_optimize:
                    share.adaptive_weight = (share.mask2d.sum()/(share.mask2d.shape[0])**2).item() if share.use_adaptive else 1
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    adaptive_gs = guidance_scale
                    noise_pred = noise_pred_uncond + adaptive_gs * (noise_pred_text - noise_pred_uncond)

                if do_corr_guidance_iter:
                    if cg_grad is None:
                        do_corr_guidance_iter = False
                    else:
                        cg_grad = cg_grad.to(dtype=torch.float64)
                        if cg_grad.std() > 0:
                            cg_grad = cg_grad/cg_grad.std()
                            cg_grad = cg_grad.to(dtype=torch.float16)
                        else:
                            do_corr_guidance_iter = False
                # End CorrFill

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                ### CorrFill
                if do_corr_guidance_iter:
                    # calculate sigma
                    scheduler_eta = (extra_step_kwargs['eta'] if 'eta' in extra_step_kwargs.keys() else 0)
                    prev_timestep = t - (self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
                    scheduler_var = self.scheduler._get_variance(t, prev_timestep)
                    sigma_t = scheduler_eta * scheduler_var ** (0.5)  # scheduler's std_dev_t
                    latents_left, latents_right = latents.chunk(2, dim=-1)
                    latents_right = latents_right - 0.03 * share.adaptive_weight * sigma_t * cg_grad * share.str_optimize
                    latents = torch.cat([latents_left, latents_right], dim=-1)
                ### End CorrFill

                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        ### CorrFill
        torch.cuda.empty_cache()
        self.wipe_cg_aux()
        latents = latents.detach()
        ### End CorrFill
        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, **condition_kwargs)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
