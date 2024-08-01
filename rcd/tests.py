import logging
from typing import Callable, List

import cv2
import numpy as np
import torch

from rcd.main import (
    LATENT_CHANNELS,
    LATENT_SCALE,
    TEXTURE_SIZE,
    decode,
    dtype_np_math,
    dtype_pt_gpu,
    make_projected_latents,
    update_projected_latents,
)

logger = logging.getLogger(__name__)


def match_latents(a, b, start=3, end=12):
    # verify that the output latents match the input latents at each atol
    for atol in [0.1**n for n in range(start, end)]:
        logger.info("testing with atol: %s", atol)
        assert np.allclose(a, b, atol=atol)
        logger.info("latents match at atol: %s", atol)

    # verify that the output steps match the input steps exactly
    # logger.warning("testing steps exactly")
    # assert np.all(input_latents == output_latents)
    # logger.info("steps match exactly")


def test_wall_128_latent(device, sd):
    # make 128-voxel wall
    # latent_values = np.zeros((128, 128, 1, LATENT_CHANNELS), dtype=dtype_np_math)
    # latent_steps = np.zeros((128, 128, 1, 1), dtype=dtype_np_math)

    # make a fake hit map with each voxel
    hit_map = [
        [(x // LATENT_SCALE, y // LATENT_SCALE, 0) for y in range(TEXTURE_SIZE)]
        for x in range(TEXTURE_SIZE)
    ]

    # make a noisy latent slice
    input_latents = np.random.randn(1, LATENT_CHANNELS, 128, 128).astype(dtype_np_math)

    # store the noisy latents and reproject them
    update_projected_latents(
        hit_map, input_latents, np.full((128, 128), 100, dtype=dtype_np_math)
    )
    output_latents, _ = make_projected_latents(hit_map)
    match_latents(input_latents, output_latents)


def test_wall_1024_latent(device, sd):
    # make 1024-voxel wall
    # latent_values = np.zeros((1024, 1024, 1, LATENT_CHANNELS), dtype=dtype_np_math)
    # latent_steps = np.zeros((1024, 1024, 1, 1), dtype=dtype_np_math)

    # make a fake hit map with each voxel
    hit_map = [
        [(x // LATENT_SCALE, y // LATENT_SCALE, 0) for y in range(1024)]
        for x in range(1024)
    ]

    # make a noisy latent slice
    input_latents = np.random.randn(1, LATENT_CHANNELS, 128, 128).astype(dtype_np_math)

    # store the noisy latents and reproject them
    update_projected_latents(
        hit_map, input_latents, np.full((128, 128), 100, dtype=dtype_np_math)
    )
    output_latents, _ = make_projected_latents(hit_map)
    match_latents(input_latents, output_latents)


@torch.no_grad()
def test_wall_128_image(device, sd):
    # wall-128-latent test using an input image instead of random latents
    # latent_values = np.zeros((128, 128, 1, LATENT_CHANNELS), dtype=dtype_np_math)
    # latent_steps = np.zeros((128, 128, 1, 1), dtype=dtype_np_math)

    # make a fake hit map with each voxel
    hit_map = [
        [(x // LATENT_SCALE, y // LATENT_SCALE, 0) for y in range(1024)]
        for x in range(1024)
    ]

    # load an image and encode it to latents
    input_image = cv2.imread("test.png")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_tensor = (
        torch.from_numpy(input_image)
        .to(dtype=dtype_pt_gpu)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    input_tensor = sd.image_processor.preprocess(input_tensor, height=1024, width=1024)
    latents = sd.vae.encode(input_tensor)[0]
    latents = latents.sample().cpu().numpy()

    # store the image latents and reproject them
    update_projected_latents(
        hit_map, latents, np.full((128, 128), 100, dtype=dtype_np_math)
    )
    output_latents, _ = make_projected_latents(hit_map)
    match_latents(latents, output_latents)

    # decode the image and test that it matches the input image
    output_image, _ = decode(sd, torch.from_numpy(output_latents))
    output_image = sd.image_processor.postprocess(output_image, output_type="pil")[0]
    output_image.save("test_out.png")

    output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
    output_image = output_image / 255.0
    match_latents(input_image, output_image, start=0, end=8)


@torch.no_grad()
def test_wall_1024_image(device, sd):
    # wall-1024-latent test using an input image instead of random latents
    # latent_values = np.zeros((1024, 1024, 1, LATENT_CHANNELS), dtype=dtype_np_math)
    # latent_steps = np.zeros((1024, 1024, 1, 1), dtype=dtype_np_math)

    # make a fake hit map with each voxel
    hit_map = [
        [(x // LATENT_SCALE, y // LATENT_SCALE, 0) for y in range(1024)]
        for x in range(1024)
    ]

    # load an image and encode it to latents
    input_image = cv2.imread("test.png")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_tensor = (
        torch.from_numpy(input_image)
        .to(dtype=dtype_pt_gpu)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    input_tensor = sd.image_processor.preprocess(input_tensor, height=1024, width=1024)
    latents = sd.vae.encode(input_tensor)[0]
    latents = latents.sample().cpu().numpy()

    # store the image latents and reproject them
    update_projected_latents(
        hit_map, latents, np.full((128, 128), 100, dtype=dtype_np_math)
    )
    output_latents, _ = make_projected_latents(hit_map)
    match_latents(latents, output_latents)

    # decode the image and test that it matches the input image
    output_image, _ = decode(sd, torch.from_numpy(output_latents))
    output_image = sd.image_processor.postprocess(output_image, output_type="pil")[0]
    output_image.save("test_out.png")

    output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
    output_image = output_image / 255.0
    match_latents(input_image, output_image, start=0, end=8)


def run_self_tests(tests: List[str], device: torch.device, sd: Callable):
    for test in tests:
        if test == "wall-128-latent":
            logger.info("running wall-128 test")
            test_wall_128_latent(device=device, sd=sd)
        elif test == "wall-1024-latent":
            logger.info("running wall-1024 test")
            test_wall_1024_latent(device=device, sd=sd)
        elif test == "wall-128-image":
            logger.info("running wall-128-image test")
            test_wall_128_image(device=device, sd=sd)
        elif test == "wall-1024-image":
            logger.info("running wall-1024-image test")
            test_wall_1024_image(device=device, sd=sd)
        else:
            logger.error("unknown test: %s", test)
