import numpy as np
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from PIL import Image


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusion(StableDiffusionPipeline):
    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[
            :, :, None, None
        ].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def generate(
        self,
        masks,
        prompt,
        negative_prompt="",
        height=512,
        width=2048,
        num_inference_steps=50,
        guidance_scale=7.5,
        bootstrapping=20,
        callback_pipe_pre=None,
        callback_pipe_post=None,
        callback_step_pre=None,
        callback_step_post=None,
    ):

        # get bootstrapping backgrounds
        # can move this outside of the function to speed up generation. i.e., calculate in init
        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(
            prompt, negative_prompt
        )  # [2 * len(prompts), 77, 768]

        latent = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8), device=self.device
        )

        if callback_pipe_pre is not None:
            result = callback_pipe_pre(self, latent)
            if result is not None:
                latent = result

        # Define panorama grid and get views
        noise = latent.clone().repeat(len(prompt) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                count.zero_()
                value.zero_()

                if callback_step_pre is not None:
                    result = callback_step_pre(self, latent, i)
                    if result is not None:
                        latent = result

                for h_start, h_end, w_start, w_end in views:
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(
                        len(prompt), 1, 1, 1
                    )
                    if i < bootstrapping:
                        bg = bootstrapping_backgrounds[
                            torch.randint(0, bootstrapping, (len(prompt) - 1,))
                        ]
                        bg = self.scheduler.add_noise(
                            bg, noise[:, :, h_start:h_end, w_start:w_end], t
                        )
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (
                            1 - masks_view[1:]
                        )

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeds
                    )["sample"]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.scheduler.step(
                        noise_pred, t, latent_view
                    )["prev_sample"]

                    if callback_step_post is not None:
                        result = callback_step_post(self, latents_view_denoised, i)
                        if result is not None:
                            latents_view_denoised = result

                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdims=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(
                        dim=0, keepdims=True
                    )

                # take the MultiDiffusion step
                latent = torch.where(count > 0, value / count, value)

        if callback_pipe_post is not None:
            result = callback_pipe_post(self, latent)
            if result is not None:
                latent = result

        # Img latents -> imgs
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
    return mask
