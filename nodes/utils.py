import numpy as np
import torch
from PIL import Image


def pack_latents(latents):
    if latents is None:
        return None

    return {
        "samples": torch.as_tensor(latents),
    }


def unpack_latents(latents):
    if isinstance(latents, dict) and "samples" in latents:
        return latents["samples"]

    return latents


def tensor_to_image(tensor):
    image = tensor[0].cpu().numpy()
    image = (image * 255.0).astype(np.uint8)
    return Image.fromarray(image)


def image_to_tensor(image):
    tensor = torch.as_tensor(np.array(image)) / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor
