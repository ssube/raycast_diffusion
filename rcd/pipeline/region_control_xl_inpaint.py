import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import is_compiled_module

from .region_control import get_views


def retrieve_latents(encoder_output: BaseOutput, generator):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MultiDiffusionXLInpaint(StableDiffusionXLControlNetInpaintPipeline):
    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        output = self.vae.encode(imgs)
        latents = retrieve_latents(output, generator=None) * 0.18215
        return latents

    @torch.no_grad()
    def get_random_background(self, n_samples, height, width):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[
            :, :, None, None
        ].repeat(1, 1, height, width)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def generate(
        self,
        masks: List[np.ndarray] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        control_image: Union[
            PipelineImageInput,
            List[PipelineImageInput],
        ] = None,
        masked_image_latents: torch.Tensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        bootstrapping: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
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
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        guidance_rescale: float = 0.0,
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
        skip_initial_steps: int = 0,
        diff_map: torch.Tensor = None,
        **kwargs,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

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

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            control_image,
            mask_image,
            strength,
            num_inference_steps,
            callback_steps,
            output_type,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            padding_mask_crop=padding_mask_crop,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start
        self._interrupt = False

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

        # 3. Encode input prompt
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
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
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
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(
                mask_image, width, height, pad=padding_mask_crop
            )
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image
        init_image = self.image_processor.preprocess(
            image,
            height=height,
            width=width,
            crops_coords=crops_coords,
            resize_mode=resize_mode,
        )
        init_image = init_image.to(dtype=original_image.dtype)

        # 5.2 Prepare control images
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                crops_coords=crops_coords,
                resize_mode=resize_mode,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
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
                    crops_coords=crops_coords,
                    resize_mode=resize_mode,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            raise ValueError(f"{controlnet.__class__} is not supported.")

        # 5.3 Prepare mask
        self.mask_processor.config.do_binarize = False
        mask = self.mask_processor.preprocess(
            mask_image,
            height=height,
            width=width,
            resize_mode=resize_mode,
            crops_coords=crops_coords,
        )

        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if denoising_start is None else False
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
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if (
                num_channels_latents + num_channels_mask + num_channels_masked_image
                != self.unet.config.in_channels
            ):
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
        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
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

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 11. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

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

        # 11.1 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # Apply the pre hook
        if callback_pipe_pre is not None:
            result = callback_pipe_pre(self, latents=latents)
            if result is not None:
                latents = result

        # Set up multi-diffusion backgrounds
        bootstrapping_backgrounds = self.get_random_background(
            bootstrapping, height, width
        )
        bg_noise = latents.clone()[0].repeat(batch_size - 1, 1, 1, 1)
        views = get_views(height, width, window_size=128, stride=16)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # preparations for diff diff
        # if original_image is not None:
        #     original_with_noise = latents.clone()
        #     thresholds = torch.arange(total_time_steps, dtype=diff_map.dtype) / total_time_steps
        #     thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(device)
        #     diff_masks = diff_map > (thresholds + (denoising_start or 0))

        # end diff diff preparations

        self._num_timesteps = len(timesteps)
        # print("num timesteps: ", self._num_timesteps, timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < skip_initial_steps:
                    progress_bar.update()
                    continue

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

                for h_start, h_end, w_start, w_end in views:
                    control_view = control_image[
                        :, :, h_start * 8 : h_end * 8, w_start * 8 : w_end * 8
                    ]
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latents[:1, :, h_start:h_end, w_start:w_end].repeat(
                        batch_size, 1, 1, 1
                    )  # using :1 and repeat blends the layers on every step, while using : keeps them separate for sharp edges
                    if i < bootstrapping:
                        bg = bootstrapping_backgrounds[
                            torch.randint(0, bootstrapping, (batch_size - 1,))
                        ]
                        # print(bg.shape, bg_noise[:, :, h_start:h_end, w_start:w_end].shape, t.shape)
                        bg = self.scheduler.add_noise(
                            bg,
                            bg_noise[:, :, h_start:h_end, w_start:w_end],
                            torch.tensor([t]),
                        )
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (
                            1 - masks_view[1:]
                        )

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latent_view] * 2)
                        if self.do_classifier_free_guidance
                        else latent_view
                    )

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    if num_channels_unet == 9:
                        latent_model_input = torch.cat(
                            [latent_model_input, mask, masked_image_latents], dim=1
                        )

                    # predict the noise residual
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }

                    # controlnet(s) inference
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latent_view
                        control_model_input = self.scheduler.scale_model_input(
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

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_view,
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

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
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

                    if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latent_view.dtype
                    latent_view_denoised = self.scheduler.step(
                        noise_pred,
                        t,
                        latent_view,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                    if latent_view_denoised.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latent_view_denoised = latent_view_denoised.to(
                                latents_dtype
                            )

                    if num_channels_unet == 4:
                        init_latents_proper = image_latents[
                            :, :, h_start:h_end, w_start:w_end
                        ]
                        if self.do_classifier_free_guidance:
                            init_mask, _ = mask.chunk(2)
                        else:
                            init_mask = mask

                        # threshold the init_mask based on the current timestep
                        step_threshold = 1 - (i / len(timesteps))
                        init_mask = init_mask >= step_threshold
                        init_mask = init_mask.to(latents.device, latents.dtype)
                        print(
                            "step %s, threshold %s, picked %s of %s texels"
                            % (
                                i,
                                step_threshold,
                                init_mask.sum().item(),
                                init_mask.numel(),
                            )
                        )

                        init_mask = init_mask[:, :, h_start:h_end, w_start:w_end]
                        noise_view = noise[:, :, h_start:h_end, w_start:w_end]

                        if i < len(timesteps) - 1:
                            noise_timestep = timesteps[i + 1]
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_proper,
                                noise_view,
                                torch.tensor([noise_timestep]),
                            )

                        latent_view_denoised = (
                            1 - init_mask
                        ) * init_latents_proper + init_mask * latent_view_denoised

                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latent_view_denoised * masks_view
                    ).sum(dim=0, keepdims=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(
                        dim=0, keepdims=True
                    )

                # take the MultiDiffusion step
                latents = torch.where(count > 0, value / count, value)

                # this needs to come after the averaging
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
                        "negative_pooled_prompt_embeds",
                        negative_pooled_prompt_embeds,
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_outputs.pop(
                        "add_neg_time_ids", add_neg_time_ids
                    )
                    mask = callback_outputs.pop("mask", mask)
                    masked_image_latents = callback_outputs.pop(
                        "masked_image_latents", masked_image_latents
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

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
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

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

        image = self.image_processor.postprocess(image, output_type=output_type)

        if padding_mask_crop is not None:
            image = [
                self.image_processor.apply_overlay(
                    mask_image, original_image, i, crops_coords
                )
                for i in image
            ]

        # Offload all models
        self.maybe_free_model_hooks()

        return image[0]
