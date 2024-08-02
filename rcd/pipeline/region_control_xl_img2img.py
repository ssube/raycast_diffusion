from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.torch_utils import is_compiled_module

from .region_control import get_views
from .region_control_xl_inpaint import retrieve_latents


class MultiDiffusionXLImg2Img(StableDiffusionXLControlNetImg2ImgPipeline):
    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        output = self.vae.encode(imgs)
        latents = retrieve_latents(output, generator=None) * 0.18215
        return latents

    @torch.no_grad()
    def get_random_background(self, n_samples, height, width):
        if n_samples == 0:
            return None

        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[
            :, :, None, None
        ].repeat(1, 1, height, width)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    def get_timesteps(
        self, num_inference_steps, strength, device, denoising_start=None
    ):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(
        self,
        masks: List[np.ndarray] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        bootstrapping: int = 20,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        callback_pipe_pre: Optional[Callable] = None,
        callback_pipe_post: Optional[Callable] = None,
        callback_step_pre: Optional[Callable] = None,
        callback_step_post: Optional[Callable] = None,
        original_image: Optional[PipelineImageInput] = None,
        diff_map: torch.FloatTensor = None,
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 1
            )
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            control_image,
            strength,
            num_inference_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3.1. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 3.2 Encode ip_adapter_image
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image and controlnet_conditioning_image
        image = self.image_processor.preprocess(image, height=height, width=width).to(
            dtype=torch.float32
        )

        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = control_image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
            height, width = control_image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # begin diff diff change
        total_time_steps = num_inference_steps
        # end diff diff change
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=(
                denoising_start if denoising_value_valid(denoising_start) else None
            ),
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        self._num_timesteps = len(timesteps)

        add_noise = True if denoising_start is None else False

        # 6. Prepare latent variables
        if latents is None:
            latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                add_noise,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 7.2 Prepare added time ids & embeddings
        if isinstance(control_image, list):
            original_size = original_size or control_image[0].shape[-2:]
        else:
            original_size = original_size or control_image.shape[-2:]
        target_size = target_size or (height, width)

        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size
        add_text_embeds = pooled_prompt_embeds

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_neg_time_ids = add_neg_time_ids.repeat(
                batch_size * num_images_per_prompt, 1
            )
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # 8.1 Apply denoising_end
        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # Set up multi-diffusion backgrounds
        bootstrapping_backgrounds = self.get_random_background(
            bootstrapping, height, width
        )
        bg_noise = latents.clone()[0].repeat(batch_size - 1, 1, 1, 1)
        views = get_views(height, width, window_size=128, stride=64)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # Create a scheduler for each view to allow for different internal state
        view_schedulers = [
            self.scheduler.__class__.from_config(self.scheduler.config)
            for _ in range(len(views))
        ]

        for scheduler in view_schedulers:
            scheduler.set_timesteps(total_time_steps, device=device)

        # preparations for diff diff
        # if original_image is not None:
        #     original_with_noise = latents.clone()
        #     thresholds = torch.arange(total_time_steps, dtype=diff_map.dtype) / total_time_steps
        #     thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(device)
        #     diff_masks = diff_map > (thresholds + (denoising_start or 0))

        # end diff diff preparations

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count.zero_()
                value.zero_()

                # diff diff
                # if original_image is not None:
                #     if i == 0 and denoising_start is None:
                #         latents = original_with_noise # [:1]
                #     else:
                #         step_mask = diff_masks[i].unsqueeze(0)
                #         # cast mask to the same type as latents etc
                #         step_mask = step_mask.to(latents.dtype)
                #         step_mask = step_mask.unsqueeze(1)  # fit shape
                #         print("blending latents with mean step mask: ", step_mask.mean())
                #         latents = original_with_noise * step_mask + latents * (1 - step_mask)

                # end diff diff

                if callback_step_pre is not None:
                    result = callback_step_pre(self, latents, i, t)
                    if result is not None:
                        latents = result

                for vi, (h_start, h_end, w_start, w_end) in enumerate(views):
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latents[:1, :, h_start:h_end, w_start:w_end].repeat(
                        batch_size, 1, 1, 1
                    )
                    # this repeat helps merge the unet passes since each prompt is working from the same latent layer
                    if i < bootstrapping:
                        bg = bootstrapping_backgrounds[
                            torch.randint(0, bootstrapping, (batch_size - 1,))
                        ]
                        bg = view_schedulers[vi].add_noise(
                            bg[:, :, h_start:h_end, w_start:w_end],
                            bg_noise[:, :, h_start:h_end, w_start:w_end],
                            torch.tensor([t]),
                        )
                        masks_view = masks_view * (1 - strength)
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (
                            1 - masks_view[1:]
                        )

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latent_view] * 2)
                        if self.do_classifier_free_guidance
                        else latent_view
                    )
                    latent_model_input = view_schedulers[vi].scale_model_input(
                        latent_model_input, t
                    )

                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }

                    # controlnet(s) inference
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latent_view
                        control_model_input = view_schedulers[vi].scale_model_input(
                            control_model_input, t
                        )
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        controlnet_added_cond_kwargs = {
                            "text_embeds": add_text_embeds.chunk(2)[1],
                            "time_ids": add_time_ids.chunk(2)[1],
                        }
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds
                        controlnet_added_cond_kwargs = added_cond_kwargs

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [
                            c * s
                            for c, s in zip(
                                controlnet_conditioning_scale, controlnet_keep[i]
                            )
                        ]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    control_image_view = control_image[
                        :, :, h_start * 8 : h_end * 8, w_start * 8 : w_end * 8
                    ]
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image_view,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )

                    if guess_mode and self.do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [
                            torch.cat([torch.zeros_like(d), d])
                            for d in down_block_res_samples
                        ]
                        mid_block_res_sample = torch.cat(
                            [
                                torch.zeros_like(mid_block_res_sample),
                                mid_block_res_sample,
                            ]
                        )

                    if (
                        ip_adapter_image is not None
                        or ip_adapter_image_embeds is not None
                    ):
                        added_cond_kwargs["image_embeds"] = image_embeds

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latent_view_denoised = view_schedulers[vi].step(
                        noise_pred,
                        t,
                        latent_view,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]

                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latent_view_denoised * masks_view
                    ).sum(dim=0, keepdims=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(
                        dim=0, keepdims=True
                    )

                # take the MultiDiffusion step
                latents = torch.where(count > 0, value / count, value)

                # call the new callbacks
                if callback_step_post is not None:
                    result = callback_step_post(self, latents, i, t)
                    if result is not None:
                        latents = result

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    add_text_embeds = callback_outputs.pop(
                        "add_text_embeds", add_text_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_outputs.pop(
                        "add_neg_time_ids", add_neg_time_ids
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        # Apply the post callback
        if callback_pipe_post is not None:
            result = callback_pipe_post(self, latents=latents)
            if result is not None:
                latents = result

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents[:1, :, :, :], return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            return latents

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return image[0]

    def generate(
        self,
        *args,
        **kwargs,
    ):
        return self(*args, **kwargs)
