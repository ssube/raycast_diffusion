import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from .utils import pack_latents, unpack_latents

FLOAT3_LIMIT = 10000.0


class Float3:
    # input: X
    # input: Y
    # input: Z
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -FLOAT3_LIMIT,
                        "max": FLOAT3_LIMIT,
                        "step": 0.01,
                    },
                ),
                "y": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -FLOAT3_LIMIT,
                        "max": FLOAT3_LIMIT,
                        "step": 0.01,
                    },
                ),
                "z": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -FLOAT3_LIMIT,
                        "max": FLOAT3_LIMIT,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT3",)

    FUNCTION = "make_float3"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def make_float3(self, x, y, z):
        return {"result": [[x, y, z]], "ui": {"float3": [x, y, z]}}


class ToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("ANY,STRING,FLOAT,INT,BOOL,LATENT,IMAGE", {}),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "to_string"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def to_string(self, input):
        return {"result": [str(input)], "ui": {"string": str(input)}}


class ShowNumpyImage:
    # input: Image
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "show_numpy_image"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def show_numpy_image(self, image):
        print("showing numpy image", image.shape, np.max(image), np.min(image))

        if np.max(image) <= 1.0:
            image = image * 255.0

        # if last dimension is 1, remove it
        if image.shape[-1] == 1:
            image = image.squeeze(2)

        # convert to PIL
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)

        # comfy image saving
        filename_prefix = "ComfyUI"
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, pil_image.width, pil_image.height
            )
        )

        filename_with_batch_num = filename.replace("%batch_num%", str(0))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        pil_image.save(
            os.path.join(full_output_folder, file), compress_level=self.compress_level
        )
        print(f"Saved {file}")
        results = [{"filename": file, "subfolder": subfolder, "type": self.type}]

        return {"result": results, "ui": {"images": results}}


class SaveImageStack:
    # input: Image Stack
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE_STACK", {}),
            },
            "optional": {
                "materials": ("MATERIALS", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "save_image_stack"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def save_image_stack(self, images, materials=None):
        # convert to PIL
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        print("saving image stack", images.shape, np.max(images), np.min(images))
        batch = np.max(images) + 1

        results = []
        for i in range(batch):
            image = images == i
            image = image.squeeze(2)  # having a single channel confuses PIL

            if materials and i > 0 and (i - 1) < len(materials.materials):
                material = materials.materials[i - 1]
                color = material.display or material.source
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, axis=2)
                image = image * color

            if np.max(image) <= 1.0:
                image = image * 255.0

            image = image.astype(np.uint8)
            pil_image = Image.fromarray(image)

            # comfy image saving
            filename_prefix = "ComfyUI"
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix, self.output_dir, pil_image.width, pil_image.height
                )
            )

            filename_with_batch_num = filename.replace("%batch_num%", str(i))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            pil_image.save(
                os.path.join(full_output_folder, file),
                compress_level=self.compress_level,
            )
            print(f"Saved {file}")

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )

        return {"result": results, "ui": {"images": results}}


class ChangeDType:
    # input: Tensor
    # input: Dtype
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensors": ("IMAGE", {}),
                "dtype": (["uint8", "float16", "float32"], {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "change_dtype"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def convert_tensor(self, tensor, dtype):
        if dtype == "uint8":
            return tensor.to(torch.uint8)
        elif dtype == "float16":
            return tensor.to(torch.float16)
        elif dtype == "float32":
            return tensor.to(torch.float32)
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

    def change_dtype(self, tensors, dtype):
        # TODO: check if tensors is a dict with samples or a list
        converted_tensors = [self.convert_tensor(t, dtype) for t in tensors]

        return {
            "result": [converted_tensors],
            "ui": {"tensor_shapes": [t.shape for t in converted_tensors]},
        }


class ChangeDevice:
    # input: Tensor
    # input: Device
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("LATENT", {}),
                "device": (["cpu", "cuda:0", "cuda:1"], {}),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "change_device"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def change_device(self, tensor, device):
        # device = torch.device(device)
        tensor = unpack_latents(tensor)
        shape = tensor.shape
        tensor = tensor.to(device)
        tensor = pack_latents(tensor)
        return {"result": [tensor], "ui": {"device": [device], "shape": [list(shape)]}}


class NormalizeArray:
    # input: Array
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "array": ("IMAGE", {}),
                "single_value": (["black", "white", "red"], {"default": "red"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "normalize_array"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/utility"

    def normalize_array(self, array, single_value):
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()

        min_val = np.min(array)
        max_val = np.max(array)

        if np.allclose(min_val, max_val):
            # special case: if the image is actually all 0 (black)
            if max_val == 0:
                pass
            elif single_value == "black":
                array = np.zeros_like(array)
            elif single_value == "white":
                array = np.ones_like(array)
            elif single_value == "red":
                shape = (*array.shape, 3)
                print("shape", shape)
                array = np.zeros(shape)
                array[..., 0] = 1.0
        else:
            array = (array - min_val) / (max_val - min_val)

        return {"result": [array], "ui": {"array": [array.shape]}}


class ExamineInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("ANY,STRING,FLOAT,INT,BOOL,LATENT,IMAGE",),
            },
        }

    RETURN_TYPES = (
        "ANY,STRING,FLOAT,INT,BOOL,LATENT,IMAGE",
        "STRING",
        "STRING",
    )

    FUNCTION = "examine_input"

    OUTPUT_NODE = True

    CATEGORY = "raycast_diffusion/utility"

    def examine_input(self, input):
        input_type = type(input).__name__
        print("examine_input", input_type)

        if isinstance(input, (str, int, float, bool)):
            input_extra = str(input)
        elif isinstance(input, list):
            input_extra = len(input)
        elif isinstance(input, dict):
            input_extra = list(input.keys())
        elif isinstance(input, torch.Tensor):
            input_extra = input.shape
        elif isinstance(input, np.ndarray):
            input_extra = input.shape
        else:
            input_extra = ""

        return {
            "result": [input, input_type, input_extra],
            "ui": {"input_type": [input_type], "input_extra": [input_extra]},
        }
