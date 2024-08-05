import argparse
from dataclasses import dataclass as numpy_dataclass
from json import dumps
from logging import INFO, basicConfig, getLogger
from os import environ, path
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoencoderTiny,
    ControlNetModel,
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from yaml import Loader, load

from .pipeline.region_control import MultiDiffusion

# from .pipeline.region_control_xl import MultiDiffusionXL
from .pipeline.region_control_xl_img2img import MultiDiffusionXLImg2Img
from .pipeline.region_control_xl_inpaint import MultiDiffusionXLInpaint
from .utils.blur import lighten_blur
from .utils.latent_correction import correction, correction_callback
from .utils.slerp import slerp

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass as dataclass  # noqa

torch.set_float32_matmul_precision("high")


logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

# start the debugger, if needed
if environ.get("DEBUG", "false").lower() in ["true", "1", "yes", "t", "y"]:
    import debugpy

    debugpy.listen(5679)
    logger.warning("waiting for debugger to attach...")
    debugpy.wait_for_client()


def parse_args():
    # keep this for something else
    # parser = argparse.ArgumentParser(description="Random Character Designer")

    parser = argparse.ArgumentParser(description="Raycast Diffusion")
    parser.add_argument(
        "--material-data",
        type=str,
        help="The material data file to use for the world",
    )
    parser.add_argument(
        "--profile-data",
        type=str,
        help="The profile data file to use for diffusion",
    )
    parser.add_argument(
        "--source-texture",
        type=str,
        help="The source texture to build the world from",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The path to the diffusion model checkpoint",
        default="1.5",
    )
    parser.add_argument(
        "--lora",
        nargs="*",
        type=str,
        help="The LoRA checkpoints to load",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="The parameter profile to use for diffusion",
    )
    parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Use the XL diffusion model",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        help="The height of the visualizer windows",
        default=480,
    )
    parser.add_argument(
        "--window-width",
        type=int,
        help="The width of the visualizer windows",
        default=640,
    )
    parser.add_argument(
        "--ceiling-material",
        type=str,
        help="The material index to use for the ceiling",
    )
    parser.add_argument(
        "--floor-material",
        type=str,
        help="The material index to use for the floor",
    )
    parser.add_argument(
        "--world-latents",
        type=str,
        help="The path to the world latents file",
    )
    parser.add_argument(
        "--world-steps",
        type=str,
        help="The path to the world steps file",
    )
    parser.add_argument(
        "--render-angle",
        type=int,
        help="The angle increment to pre-render",
        default=0,
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Only load the decoder model (greatly reduces VRAM usage)",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        help="The pipeline compiler to use",
        choices=["torch", "sfast", "onediff"],
    )
    parser.add_argument(
        "--export-geometry",
        action="store_true",
        help="Export the world geometry to files",
    )
    parser.add_argument(
        "--splat-texture",
        type=str,
        help="Splat a texture onto the world",
        default="splat.png",
    )
    parser.add_argument(
        "--self-test",
        type=str,
        nargs="*",
        help="Run the following self-tests",
        choices=[
            "wall-128-latent",
            "wall-1024-latent",
            "wall-128-image",
            "wall-1024-image",
        ],
    )
    parser.add_argument(
        "--camera-path",
        type=str,
        help="The path to the camera trajectory file",
    )
    parser.add_argument(
        "--exit-after-path",
        action="store_true",
        help="Exit after rendering the camera path",
    )
    return parser.parse_args()


# region: Dataclasses
@dataclass
class MaterialData:
    name: str
    source: Tuple[int, int, int]
    prompt: str
    height: int
    display: Tuple[int, int, int] | None = None
    negative_prompt: str | None = None
    start_height: int | None = None
    # over/under materials
    material_over: str | None = None
    material_under: str | None = None


@dataclass
class BackgroundMaterialData:
    prompt: str
    negative_prompt: str | None = None


@dataclass
class MaterialFile:
    background: BackgroundMaterialData
    materials: List[MaterialData]


@dataclass
class ProfileData:
    name: str
    cfg: float
    steps: int
    bootstrapping_steps: int
    capture_steps: int
    width: int
    height: int


@dataclass
class ProfileFile:
    profiles: List[ProfileData]


@numpy_dataclass
class WorldVolume:
    latents: np.ndarray
    steps: np.ndarray
    voxels: np.ndarray


@numpy_dataclass
class ProjectionData:
    index: np.ndarray
    latents: np.ndarray
    steps: np.ndarray


@numpy_dataclass
class RaycastData:
    points: np.ndarray
    depth: np.ndarray


@numpy_dataclass
class RenderData:
    index: np.ndarray
    depth: np.ndarray


@numpy_dataclass
class DiffusionData:
    image: np.ndarray
    latents: np.ndarray
    steps: np.ndarray
    intermediate_latents: np.ndarray | None


@numpy_dataclass
class GeometryData:
    colors: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    vertices: np.ndarray


@numpy_dataclass
class CameraData:
    mesh_vis: o3d.visualization.Visualizer
    raycast_scene: Any


# endregion

# region: Constants
CUBE_VERTICES = [
    [0, 0, 0],  # 0
    [0, 0, 1],  # 1
    [0, 1, 1],  # 2
    [0, 1, 0],  # 3
    [1, 0, 0],  # 4
    [1, 0, 1],  # 5
    [1, 1, 1],  # 6
    [1, 1, 0],  # 7
]

FACE_NORMALS = [
    [0, 1, 0],  # top (Y+)
    [0, 0, 1],  # south (Z+)
    [1, 0, 0],  # east (X+)
    [0, 0, -1],  # north (Z-)
    [-1, 0, 0],  # west (X-)
    [0, -1, 0],  # bottom (Y-)
]

FACE_VERTICES = [
    # Top face (Y+)
    [
        [
            2,
            3,
            7,
        ],
        [
            2,
            7,
            6,
        ],
    ],
    # South face (Z+)
    [
        [
            1,
            2,
            6,
        ],
        [
            1,
            6,
            5,
        ],
    ],
    # East face (X+)
    [
        [
            5,
            6,
            7,
        ],
        [
            5,
            7,
            4,
        ],
    ],
    # North face (Z-)
    [
        [
            4,
            7,
            3,
        ],
        [
            4,
            3,
            0,
        ],
    ],
    # West face (X-)
    [
        [
            0,
            3,
            2,
        ],
        [
            0,
            2,
            1,
        ],
    ],
    # Bottom face (Y-)
    [
        [
            0,
            1,
            5,
        ],
        [
            0,
            5,
            4,
        ],
    ],
]

LATENT_CHANNELS = 4
LATENT_SCALE = 8

TEXTURE_SIZE = 1024
LATENT_SIZE = TEXTURE_SIZE // LATENT_SCALE
# endregion

# region: Config
# things that should be config
dtype_pt_cpu = torch.float32
dtype_pt_gpu = torch.float32
dtype_np_index = np.uint8
dtype_np_math = np.float32
interpolation_mode = "bilinear"
path_latents = "latents.npy"
path_steps = "steps.npy"
preview_steps = 5
voxel_checkerboard = 0.85
voxel_size = 1.0
# endregion

# region: Globals
# depth_texture: Optional[np.ndarray] = None
# diffusion_texture: Optional[np.ndarray] = None
# latent_steps: Optional[np.ndarray] = None
# latent_values: Optional[np.ndarray] = None
# material_data: Optional[MaterialFile] = None
# profile_data: Optional[ProfileData] = None
# projected_texture: Optional[np.ndarray] = None
# ray_depth: Optional[np.ndarray] = None
# ray_points: Optional[np.ndarray] = None
# world_voxels: Optional[np.ndarray] = None
# endregion


# region: IO
def load_material_data(material_path: str) -> MaterialFile:
    logger.info(f"loading material data from {material_path}")
    with open(material_path, "r") as f:
        data = load(f, Loader=Loader)

    return MaterialFile(**data)


def load_profile_data(profile_path: str) -> ProfileFile:
    logger.info(f"loading profile data from {profile_path}")
    with open(profile_path, "r") as f:
        data = load(f, Loader=Loader)

    return ProfileFile(**data)


def load_source_texture(source_texture: str) -> np.ndarray:
    logger.info(f"loading source texture from {source_texture}")
    img = cv2.imread(source_texture)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_latents(
    world: WorldVolume,
    require_existing: bool = False,
    always_init: bool = False,
    seed=None,
) -> WorldVolume:
    logger.info("loading latents")

    generator = np.random.default_rng(seed)

    if path.exists(path_latents) and not always_init:
        logger.info("loading latents from disk: %s", path_latents)
        latent_values = np.load(path_latents)
    elif require_existing:
        raise FileNotFoundError(f"latents not found at {path_latents}")
    else:
        logger.info("creating new latents")
        latent_values = generator.normal(
            size=(*world.voxels.shape, LATENT_CHANNELS)
        ).astype(dtype_np_math)

    if path.exists(path_steps) and not always_init:
        logger.info("loading steps from disk: %s", path_steps)
        latent_steps = np.load(path_steps)
    elif require_existing:
        raise FileNotFoundError(f"steps not found at {path_steps}")
    else:
        logger.info("creating new steps")
        latent_steps = np.zeros((*world.voxels.shape, 1), dtype=dtype_np_math)

    logger.info("loaded latents with dimensions: %s", latent_values.shape)
    logger.info("loaded steps with dimensions: %s", latent_steps.shape)

    # verify that the latents are the correct size
    if latent_values.shape != (*world.voxels.shape, LATENT_CHANNELS):
        raise ValueError("latent dimensions do not match world voxels")

    if latent_steps.shape != (*world.voxels.shape, 1):
        raise ValueError("steps dimensions do not match world voxels")

    return WorldVolume(latents=latent_values, steps=latent_steps, voxels=world.voxels)


def save_latents(world: WorldVolume):
    logger.info("saving latents")

    if world.latents is not None:
        logger.info("saving latents with dimensions: %s", world.latents.shape)
        np.save(path_latents, world.latents)

    if world.steps is not None:
        logger.info("saving steps with dimensions: %s", world.steps.shape)
        np.save(path_steps, world.steps)


# endregion


# region: Geometry
def apply_material_stack(
    materials: MaterialFile,
    material_index: int,
    world_volume: np.ndarray,
    x: int,
    y: int,
    min_height=0,
) -> None:
    """
    Apply a material stack to the world volume.
    """

    world_index = material_index + 1
    material = materials.materials[material_index]

    # apply the material to the world volume
    start_height = material.start_height or min_height
    for z in range(start_height, material.height):
        if world_volume[x, y, z] == 0:
            world_volume[x, y, z] = world_index
        else:
            logger.debug(f"voxel {x}, {y}, {z} already has a material")

    # apply the over material
    if material.material_over:
        over_index = next(
            i
            for i, m in enumerate(materials.materials)
            if m.name == material.material_over
        )
        apply_material_stack(
            materials, over_index, world_volume, x, y, min_height=material.height
        )

    # apply the under material
    if material.material_under:
        under_index = next(
            i
            for i, m in enumerate(materials.materials)
            if m.name == material.material_under
        )
        apply_material_stack(materials, under_index, world_volume, x, y)


def make_world(
    materials: MaterialFile,
    source_texture: np.ndarray,
    ceiling_material: str | None = None,
    floor_material: str | None = None,
    voxel_multiplier: int = 1,
) -> WorldVolume:
    """
    Create a 3D voxel grid from the source texture and assign materials based on the material data.
    """

    logger.info("building world voxels from source texture: %s", source_texture.shape)

    world_width, world_depth, _channels = source_texture.shape
    world_height = max(m.height for m in materials.materials)

    volume = np.zeros((world_width, world_depth, world_height), dtype=dtype_np_index)

    for x in range(source_texture.shape[0]):
        for y in range(source_texture.shape[1]):
            source_material = source_texture[x, y]
            for i, m in enumerate(materials.materials):
                if (source_material == m.source).all():
                    apply_material_stack(materials, i, volume, x, y)

    # add ceiling and floor to voxels that do not have a material
    if floor_material is not None:
        floor_index = next(
            i for i, m in enumerate(materials.materials) if m.name == floor_material
        )
        for x in range(source_texture.shape[0]):
            for y in range(source_texture.shape[1]):
                bottom = volume[x, y, 0]
                if bottom == 0:
                    volume[x, y, 0] = floor_index + 1

    if ceiling_material is not None:
        ceiling_index = next(
            i for i, m in enumerate(materials.materials) if m.name == ceiling_material
        )
        for x in range(source_texture.shape[0]):
            for y in range(source_texture.shape[1]):
                top = volume[x, y, -1]
                if top == 0:
                    volume[x, y, -1] = ceiling_index + 1

    if voxel_multiplier > 1:
        volume = np.kron(
            volume,
            np.ones((voxel_multiplier, voxel_multiplier, voxel_multiplier)).astype(
                dtype_np_index
            ),
        )

    return WorldVolume(
        latents=np.zeros((*volume.shape, LATENT_CHANNELS), dtype=dtype_np_math),
        steps=np.zeros((*volume.shape, 1), dtype=dtype_np_math),
        voxels=volume,
    )


def make_world_triangles(world: WorldVolume, materials: MaterialFile) -> GeometryData:
    """
    Create a triangle mesh with colors and normals from the world volume.
    """

    logger.info("building world triangles from world volume: %s", world.voxels.shape)

    colors = []
    normals = []
    triangles = []
    vertices = []

    x_max = world.voxels.shape[0]
    y_max = world.voxels.shape[1]
    z_max = world.voxels.shape[2]

    face_normals = list(enumerate(FACE_NORMALS))

    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                if not world.voxels[x, y, z]:
                    # skip empty voxels. the filled voxels are responsible for the mesh faces
                    continue

                # for each face of the voxel
                for i, face in face_normals:
                    # clamp the face to the bounds of the world volume
                    fx = x + face[0]
                    fy = y + face[1]
                    fz = z + face[2]
                    fx = 0 if fx < 0 else (x_max - 1) if fx >= x_max else fx
                    fy = 0 if fy < 0 else (y_max - 1) if fy >= y_max else fy
                    fz = 0 if fz < 0 else (z_max - 1) if fz >= z_max else fz

                    voxel = world.voxels[x, y, z]
                    neighbor = world.voxels[fx, fy, fz]

                    # check if the face has a material boundary (the next voxel has a different material)
                    if voxel != neighbor:
                        # add a face to the mesh
                        material = materials.materials[voxel - 1]
                        color = material.display or material.source
                        color = [c / 255.0 for c in color]

                        # tweak the color slightly to make a checkerboard pattern
                        if (x + y + z) % 2 == 0:
                            color = [c * voxel_checkerboard for c in color]

                        colors.extend([color] * 6)
                        normals.extend([face] * 6)

                        # get origin for the voxel
                        ox = x * voxel_size
                        oy = y * voxel_size
                        oz = z * voxel_size

                        # add the vertices for each triangle in the face
                        for triangle in FACE_VERTICES[i]:
                            indices = list(triangle)
                            indices.reverse()
                            for vertex in indices:
                                offset = CUBE_VERTICES[vertex]
                                vertices.append(
                                    [
                                        ox + offset[0] * voxel_size,
                                        oy + offset[1] * voxel_size,
                                        oz + offset[2] * voxel_size,
                                    ]
                                )

                        # build triangles using the vertex indices
                        triangles.append(
                            [len(vertices) - 6, len(vertices) - 5, len(vertices) - 4]
                        )
                        triangles.append(
                            [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]
                        )

    return GeometryData(
        colors=np.array(colors),
        normals=np.array(normals),
        triangles=np.array(triangles),
        vertices=np.array(vertices),
    )


def project_voxel_hit_map(raycast: RaycastData) -> List[List[Tuple[int, int, int]]]:
    """
    Create a texture from the hit voxels.
    """
    hit_voxels = raycast.points
    width, height, _ = hit_voxels.shape

    hit_map = [[None for _ in range(height)] for _ in range(width)]

    for x in range(width):
        for y in range(height):
            voxel = hit_voxels[x, y]
            if (voxel > 0).all():
                hit_map[x][y] = tuple(list(voxel))

    return hit_map


def make_projected_texture(
    hit_map, world: WorldVolume, materials: MaterialFile
) -> np.ndarray:
    """
    Create a texture from the hit voxels.
    """
    width = len(hit_map)
    height = len(hit_map[0])
    texture = np.zeros((width, height, 3), dtype=dtype_np_index)

    for x in range(width):
        for y in range(height):
            voxel = hit_map[x][y]
            if voxel is not None:
                material_index = world.voxels[voxel]
                if material_index:
                    material = materials.materials[material_index - 1]
                    texture[x, y] = material.display or material.source

    return texture


def make_projected_index_texture(hit_map, world: WorldVolume) -> np.ndarray:
    """
    Create a texture from the hit voxels.
    """
    width = len(hit_map)
    height = len(hit_map[0])
    texture = np.zeros((width, height, 1), dtype=dtype_np_index)

    for x in range(width):
        for y in range(height):
            voxel = hit_map[x][y]
            if voxel is not None:
                material_index = world.voxels[voxel]
                if material_index:
                    texture[x, y] = material_index

    return texture


def make_projected_depth_texture(raycast: RaycastData) -> np.ndarray:
    """
    Create a texture from the ray depth, where the color is mapped from 0 to the maximum depth.
    """

    min_depth = np.min(raycast.depth)  # np.where(ray_depth > 0, ray_depth, np.inf))
    max_depth = np.max(raycast.depth)

    adj_ray_depth = raycast.depth - min_depth
    adj_ray_range = max_depth - min_depth
    texture = 1 - (adj_ray_depth / adj_ray_range)

    return texture


def make_projected_latents(
    hit_map, world: WorldVolume, seed: int | None = None
) -> ProjectionData:
    """
    Create a latent vector from the hit voxels.
    """

    latent_values = world.latents
    latent_steps = world.steps

    generator = np.random.default_rng(seed)

    width = len(hit_map)
    height = len(hit_map[0])
    noise = generator.normal(
        size=(1, LATENT_CHANNELS, width // LATENT_SCALE, height // LATENT_SCALE)
    ).astype(dtype_np_math)
    latents = np.zeros(
        (1, LATENT_CHANNELS, width // LATENT_SCALE, height // LATENT_SCALE),
        dtype=dtype_np_math,
    )
    counts = np.zeros(
        (1, 1, width // LATENT_SCALE, height // LATENT_SCALE), dtype=dtype_np_math
    )
    steps = np.zeros(
        (width // LATENT_SCALE, height // LATENT_SCALE), dtype=dtype_np_math
    )

    if latent_values is None or latent_steps is None:
        load_latents(require_existing=False)

    for x in range(width):
        for y in range(height):
            voxel = hit_map[x][y]
            if voxel is not None:
                step = latent_steps[voxel][0]
                count = counts[0, :, x // LATENT_SCALE, y // LATENT_SCALE]
                if step > 0 and np.all(count == 0):
                    # accumulate and average latents for multiple voxels
                    latents[
                        0, :, x // LATENT_SCALE, y // LATENT_SCALE
                    ] += latent_values[voxel]
                    counts[0, :, x // LATENT_SCALE, y // LATENT_SCALE] += 1
                    steps[x // LATENT_SCALE, y // LATENT_SCALE] = step

    logger.debug("average voxel counts: %s", np.mean(counts))
    latents = np.where(counts > 0, latents / np.max(counts, 1), noise)

    return ProjectionData(index=None, latents=latents, steps=steps)


def update_projected_latents(
    hit_map, world: WorldVolume, results: DiffusionData
) -> int:
    width = len(hit_map)
    height = len(hit_map[0])

    # rescale from screen space to latent space without interpolating
    width_scale = results.latents.shape[2] / width
    height_scale = results.latents.shape[3] / height

    # avoid updating the same voxel multiple times
    updated = set()

    # average instead
    latent_accum = np.zeros_like(world.latents)
    count_accum = np.zeros_like(world.steps)
    step_accum = np.zeros_like(world.steps)

    for x in range(width):
        for y in range(height):
            voxel = hit_map[x][y]
            if voxel is not None and voxel not in updated:
                updated.add(voxel)

                # scale current coordinates and truncate to integer
                lx = int(x * width_scale)
                ly = int(y * height_scale)

                prev_step = world.steps[voxel]
                new_step = results.steps[lx, ly]

                # only update latents with new steps to prevent degradation of completed regions
                if new_step > prev_step:
                    # latent_values[voxel] = new_latents[
                    #    :1, :, lx, ly
                    # ]
                    world.steps[voxel] = new_step
                    count_accum[voxel] += 1
                    latent_accum[voxel] += results.latents[0, :, lx, ly]
                    step_accum[voxel] = new_step
                # else:
                #    # blend existing regions
                #    max_step = max(prev_step, new_step)
                #    latent_steps[voxel] = max_step
                #    count_accum[voxel] += 2
                #    latent_accum[voxel] += latent_values[voxel] + new_latents[
                #        0, :, lx, ly
                #    ]
                #    step_accum[voxel] = max_step

    # update the latents and steps
    world.latents = np.where(count_accum > 0, latent_accum / count_accum, world.latents)
    world.steps = np.where(count_accum > 0, step_accum, world.steps)

    # logger.info("updated %s voxels", count_accum[count_accum > 0].shape)
    logger.info("updated voxel count: %s", len(updated))

    # save the updated latents to disk
    save_latents(world)

    return len(updated)


# endregion


# region: Diffusion
class DiffusionDecoder:
    image_processor: VaeImageProcessor
    unet: None
    vae: AutoencoderTiny
    vae_scale_factor: int

    def __init__(self, vae: AutoencoderTiny):
        self.unet = None
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


def load_tiny_vae(
    repo: str | None = None,
    sdxl=False,
):
    if sdxl:
        return AutoencoderTiny.from_pretrained(
            repo or "madebyollin/taesdxl", torch_dtype=dtype_pt_gpu
        )
    else:
        return AutoencoderTiny.from_pretrained(
            repo or "madebyollin/taesd", torch_dtype=dtype_pt_gpu
        )


def load_sd(
    checkpoint: str,
    compiler: Optional[Literal["torch", "sfast", "onediff"]] = None,
    decode_only: bool = False,
    optimize: bool = True,
    quantize_unet: bool = False,
    sdxl: bool = False,
    tiny_vae: bool = True,
    xformers: bool = False,
    scheduler: str = "ddim",
) -> Callable:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("loading checkpoint %s to device %s", checkpoint, device)

    if decode_only:
        logger.warning("loading decoder only")
        vae = load_tiny_vae(checkpoint, sdxl)
        vae = vae.to(device)
        return DiffusionDecoder(vae)

    if sdxl:
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=dtype_pt_gpu
        )
        sd = MultiDiffusionXLInpaint.from_single_file(
            checkpoint, controlnet=controlnet, device=device, dtype=dtype_pt_gpu
        )
        # sd.vae = vae
        sd.watermark = None
        sd = sd.to(device)
    else:
        sd = MultiDiffusion.from_single_file(
            checkpoint, device=device, dtype=dtype_pt_gpu
        )
        # sd.vae = vae
        sd = sd.to(device)

    SCHEDULER_TYPES = {
        "ddim": (DDIMScheduler, {}),
        "dpm++2m": (DPMSolverMultistepScheduler, {}),
        "dpm++2m-karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
        "dpm++2m-sde": (
            DPMSolverMultistepScheduler,
            {"algorithm_type": "sde-dpmsolver++"},
        ),
        "dpm++2m-sde-karras": (
            DPMSolverMultistepScheduler,
            {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
        ),
        "dpm++sde": (DPMSolverSinglestepScheduler, {}),
        "dpm++sde-karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
        "dpm2": (KDPM2DiscreteScheduler, {}),
        "dpm2-karras": (KDPM2DiscreteScheduler, {"use_karras_sigmas": True}),
        "euler": (EulerDiscreteScheduler, {}),
        "euler-ancestral": (EulerAncestralDiscreteScheduler, {}),
        "heun": (HeunDiscreteScheduler, {}),
        "lcm": (LCMScheduler, {}),
        "lms": (LMSDiscreteScheduler, {}),
        "lms-karras": (LMSDiscreteScheduler, {"use_karras_sigmas": True}),
        "deis": (DEISMultistepScheduler, {}),
        "uni-pc": (UniPCMultistepScheduler, {}),
    }

    scheduler = scheduler.lower()
    if scheduler not in SCHEDULER_TYPES:
        raise ValueError(f"unknown scheduler type: {scheduler}")

    scheduler_class, scheduler_args = SCHEDULER_TYPES[scheduler]
    sd.scheduler = scheduler_class.from_config(sd.scheduler.config, **scheduler_args)

    if tiny_vae:
        logger.warning("using tiny VAE")
        vae = load_tiny_vae(checkpoint, sdxl)
        sd.vae = vae

    if xformers:
        logger.warning("enabling xformers")
        sd.enable_xformers_memory_efficient_attention()

    if optimize:
        logger.warning("enabling SDXL optimizations")
        sd.enable_vae_slicing()
        sd.enable_vae_tiling()
        sd.unet.to(memory_format=torch.channels_last)

    if compiler == "torch":
        logger.warning("compiling unet using Torch")
        sd.unet = torch.compile(sd.unet, mode="reduce-overhead", fullgraph=True)

    if compiler == "sfast":
        logger.warning("compiling unet using SFAST")
        from sfast.compilers.diffusion_pipeline_compiler import (
            CompilationConfig,
            compile,
        )

        config = CompilationConfig.Default()
        try:
            import xformers

            config.enable_xformers = True
        except ImportError:
            logger.warning("xformers not installed, skipping")
        try:
            import triton  # noqa

            config.enable_triton = True
        except ImportError:
            logger.warning("Triton not installed, skipping")

        sd = compile(sd, config)

    if compiler == "onediff":
        logger.warning("compiling unet using OneDiff")
        from onediff.infer_compiler import oneflow_compile

        sd.unet = oneflow_compile(sd.unet)

        if quantize_unet:
            from diffusers.utils import USE_PEFT_BACKEND

            assert USE_PEFT_BACKEND
            sd.unet = torch.quantization.quantize_dynamic(
                sd.unet, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )

    return sd


def preview(
    sd: Callable,
    profile: ProfileData,
    render: RenderData,
    world: WorldVolume,
    raycast: RaycastData,
    materials: MaterialFile,
    seed: int | None = None,
):
    logger.info("previewing latents")

    hit_map = project_voxel_hit_map(raycast.points)
    projection = make_projected_latents(hit_map, world, seed)
    projection.index = make_projected_index_texture(hit_map, world)

    # run final steps
    sd = MultiDiffusionXLImg2Img.from_pipe(sd)
    return run_diffusion(sd, profile, render, projection, materials)


def decode(
    sd: Callable,
    input_latents: torch.Tensor,
):
    logger.info("decoding latents: %s", input_latents.shape)

    has_latents_mean = (
        hasattr(sd.vae.config, "latents_mean")
        and sd.vae.config.latents_mean is not None
    )
    has_latents_std = (
        hasattr(sd.vae.config, "latents_std") and sd.vae.config.latents_std is not None
    )
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(sd.vae.config.latents_mean)
            .view(1, LATENT_CHANNELS, 1, 1)
            .to(input_latents.device, input_latents.dtype)
        )
        latents_std = (
            torch.tensor(sd.vae.config.latents_std)
            .view(1, LATENT_CHANNELS, 1, 1)
            .to(input_latents.device, input_latents.dtype)
        )
        input_latents = (
            input_latents * latents_std / sd.vae.config.scaling_factor + latents_mean
        )
    else:
        input_latents = input_latents / sd.vae.config.scaling_factor

    # TODO: convert to fp16 if needed
    last_image = sd.vae.decode(input_latents[:1, :, :, :], return_dict=False)[0]  # [0]
    last_image = last_image.detach()

    output_image = sd.image_processor.postprocess(last_image, output_type="pil")[0]
    output_image.save("last-decode.png")

    return last_image, output_image


def interpolate_latents(
    latents: torch.Tensor,
    size: Tuple[int, int],
    mode: str | None = None,
) -> torch.Tensor:
    logger.debug("interpolating latents: %s to %s", latents.shape, size)

    original_dims = len(latents.shape)

    if latents.shape[-2:] == size:
        return latents

    while len(latents.shape) < 4:
        latents = latents.unsqueeze(0)

    logger.warning("interpolating latents: from %s to %s", latents.shape, size)
    latents = torch.nn.functional.interpolate(
        latents, size=size, mode=(mode or interpolation_mode)
    )

    while len(latents.shape) > original_dims and latents.shape[0] == 1:
        latents = latents.squeeze(0)

    return latents


def embed_prompts(
    sd: Callable,
    prompts: List[str],
    negative_prompts: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info("embedding prompts")

    if hasattr(sd, "tokenizer_2"):
        compel = Compel(
            tokenizer=[sd.tokenizer, sd.tokenizer_2],
            text_encoder=[sd.text_encoder, sd.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        prompt_embeds, pooled_prompt_embeds = compel(prompts)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompts)
    else:
        compel = Compel(tokenizer=sd.tokenizer, text_encoder=sd.text_encoder)
        prompt_embeds = compel(prompts)
        negative_prompt_embeds = compel(negative_prompts)
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None

    return (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def make_mask_stack(
    render: RenderData,
    materials: MaterialFile,
):
    masks = []
    prompts = []
    negative_prompts = []

    # background prompt has to always be the first one, because the mask pipeline treats it differently
    mask = render.index == 0
    masks.append(mask)
    prompts.append(materials.background.prompt)
    negative_prompts.append(materials.background.negative_prompt or "")

    # add foreground masks
    for i, material in enumerate(materials.materials):
        mask = (render.index == i + 1).astype(dtype_np_math)
        # only include the material if it is used in the world (any(mask > 0))
        if np.any(mask):
            masks.append(mask)
            prompts.append(material.prompt)
            negative_prompts.append(material.negative_prompt or "")

    masks = torch.stack([torch.from_numpy(m).to(dtype=dtype_pt_gpu) for m in masks])
    masks = torch.permute(masks, (0, 3, 1, 2))
    return masks, prompts, negative_prompts


@torch.no_grad()
def run_diffusion(
    sd: Callable,
    profile: ProfileData,
    # index_texture: np.ndarray,
    # depth_texture: np.ndarray,
    render: RenderData,
    # input_latents: np.ndarray,
    # input_steps: np.ndarray,
    projection: ProjectionData,
    material_data: MaterialFile,
    callback: Callable | None = None,
    file_suffix: str = "",
    skip_steps: int = 0,
    strength: float = 1.0,
    controlnet_strength: float = 0.6,
    seed: int | None = None,
    color_correction: bool = True,
) -> DiffusionData:
    device = sd.unet.device

    logger.info(
        "running diffusion using profile %s and %s steps", profile.name, profile.steps
    )

    masks, prompts, negative_prompts = make_mask_stack(render, material_data)

    input_width = projection.latents.shape[2]
    input_height = projection.latents.shape[3]
    # input_size = (input_width, input_height)
    latent_size = (profile.width // LATENT_SCALE, profile.height // LATENT_SCALE)

    masks = interpolate_latents(masks, size=latent_size)
    masks = masks.to(device)
    # masks = lighten_blur(masks)
    # masks = lighten_blur(masks)

    input_latents = torch.from_numpy(projection.latents).to(device)
    input_latents = interpolate_latents(input_latents, size=latent_size)
    input_latents = input_latents.repeat(len(prompts), 1, 1, 1)

    input_steps = torch.from_numpy(projection.steps).to(device)
    input_steps = interpolate_latents(input_steps, size=latent_size)

    capture_step = profile.steps - profile.capture_steps
    if skip_steps > 0:  # TODO: can this be removed?
        mask_image = torch.ones(latent_size)
        mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        input_noise = input_latents
    else:
        # input_steps_large = interpolate_latents(input_steps, size=(profile.width, profile.height))
        mask_min = torch.as_tensor(0.25).to(device)
        mask_max = torch.as_tensor(1.0).to(device)
        mask_image = torch.lerp(
            mask_max, mask_min, input_steps / capture_step
        )  # capture step here makes everything noisier
        mask_image = mask_image.unsqueeze(0)
        mask_image = lighten_blur(mask_image)
        save_numpy_image(mask_image.squeeze(0).cpu().numpy(), f"mask{file_suffix}.png")

        # add noise to the input latents where steps are zero
        noise = torch.randn_like(input_latents).to(device)
        # step_mask = lighten_blur(input_steps.unsqueeze(0)).squeeze(0)
        input_noise = torch.lerp(noise, input_latents, mask_image)

        noise_image = input_noise.permute(0, 2, 3, 1)[0].cpu().numpy()
        noise_range = np.max(noise_image) - np.min(noise_image)
        noise_image = (noise_image - np.min(noise_image)) / noise_range
        save_numpy_image(noise_image, f"noise{file_suffix}.png")

    depth_image = torch.from_numpy(render.depth).to(device)
    depth_image = depth_image.permute(2, 0, 1).unsqueeze(0)

    capture_latents = None

    def pre_step(pipe, latents, step, timestep):
        """
        Inject the previous latents after the skipped steps.
        """

        if step < skip_steps:
            return latents

        # interpolate between the new latents and the input latents from step 0 up to skip_steps
        r = torch.where(input_steps >= 1, step / input_steps, 0)
        r = torch.where(r > 1, 0, r)
        logger.info(
            "blending latents at step %s with mean steps %s and mean r %s",
            step,
            input_steps.mean().item(),
            r.mean().item(),
        )
        blended_latents = latents.clone()
        blended_latents[:1] = slerp(
            latents[:1], input_latents[:1], r
        )  # only slerp the layer where blending occurs

        if color_correction:
            blended_latents = correction(blended_latents, timestep)

        return blended_latents

    def post_step(pipe, latents, step, timestep):
        """
        Capture latents before they are complete.
        """
        nonlocal capture_latents

        if step == capture_step:
            capture_latents = latents

        if callback is not None:
            callback(step, latents, timestep, profile.steps)

    # min_input_step = input_steps.min().cpu().item()
    # skip_steps = max(skip_steps, min_input_step)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = embed_prompts(sd, prompts, negative_prompts)

    generator = None
    if seed is not None:
        generator = torch.random.manual_seed(seed)

    final_latents = sd.generate(
        image=input_noise,
        mask_image=mask_image.to(dtype=dtype_pt_gpu),
        control_image=depth_image.to(dtype=dtype_pt_gpu),
        # latents=input_noise,
        masks=masks.to(dtype=dtype_pt_gpu),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        height=profile.height,
        width=profile.width,
        strength=strength,
        num_inference_steps=profile.steps,
        bootstrapping=profile.bootstrapping_steps,
        guidance_scale=profile.cfg,
        controlnet_conditioning_scale=controlnet_strength,
        output_type="latent",
        callback_step_pre=pre_step,
        callback_step_post=post_step,
        skip_initial_steps=skip_steps,
        callback_on_step_end=correction_callback,
        generator=generator,
    )

    output_image = decode(sd, final_latents)[0]
    output_image = sd.image_processor.postprocess(output_image, output_type="pil")[0]
    output_image.save(f"output{file_suffix}.png")

    if capture_latents is None:
        logger.warning(
            "no latents captured at step %s, using final latents", capture_step
        )
        capture_latents = final_latents
        capture_step = profile.steps

    output_steps = np.full(
        (input_width, input_height), capture_step, dtype=dtype_np_math
    )
    return DiffusionData(
        image=output_image,
        latents=final_latents,
        steps=output_steps,
        intermediate_latents=capture_latents,
    )


@torch.no_grad()
def run_highres(
    sd: Callable,
    profile: ProfileData,
    render: RenderData,
    projection: ProjectionData,
    previous_output: Image.Image,
    material_data: MaterialFile,
    callback: Callable | None = None,
    file_suffix: str = "",
    strength: float = 0.4,
    controlnet_strength: float = 0.6,
    seed: int | None = None,
    color_correction: bool = True,
) -> DiffusionData:
    input_width = projection.latents.shape[2]
    input_height = projection.latents.shape[3]
    latent_size = (profile.width // LATENT_SCALE, profile.height // LATENT_SCALE)

    masks, prompts, negative_prompts = make_mask_stack(render, material_data)

    # highres pass
    # TODO: remove this scale factor
    highres_scale = 1.5
    highres_size = (
        int(profile.width * highres_scale),
        int(profile.height * highres_scale),
    )
    highres_latent_size = (
        int(latent_size[0] * highres_scale),
        int(latent_size[1] * highres_scale),
    )

    sd_highres = MultiDiffusionXLImg2Img.from_pipe(sd)
    device = sd.unet.device

    masks = interpolate_latents(masks, size=highres_latent_size)
    masks = masks.to(device)

    depth_image = torch.from_numpy(render.depth).to(device)
    depth_image = depth_image.permute(2, 0, 1).unsqueeze(0)
    highres_depth = interpolate_latents(depth_image, size=highres_size)
    save_numpy_image(highres_depth.squeeze(0), f"highres_depth{file_suffix}.png")

    # TODO: attempt latent upscale for highres
    # highres_latents = upscale(final_latents, highres_scale)
    highres_image = previous_output.resize(highres_size)
    highres_image.save(f"highres_input{file_suffix}.png")

    highres_masks = interpolate_latents(masks, size=highres_latent_size)
    highres_masks = lighten_blur(highres_masks)
    for i in range(highres_masks.shape[0]):
        save_numpy_image(highres_masks[i, :, :, :], f"highres_mask{file_suffix}{i}.png")

    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = embed_prompts(sd, prompts, negative_prompts)

    def callback_wrapper(pipe, step, timestep, cbk):
        result = cbk
        if color_correction:
            result = correction_callback(pipe, step, timestep, cbk)

        latents = cbk["latents"]
        if callback is not None:
            callback(step, latents, timestep, profile.steps)

        return result

    generator = None
    if seed is not None:
        generator = torch.random.manual_seed(seed)

    highres_latents = sd_highres(
        masks=highres_masks,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        control_image=highres_depth,
        image=highres_image,
        guidance_scale=profile.cfg,
        num_inference_steps=profile.steps * 2,
        bootstrapping=0,
        return_dict=False,
        strength=strength,  # switching this from 0.4 to 0.5 makes a huge difference
        controlnet_conditioning_scale=controlnet_strength,
        output_type="latent",
        callback_on_step_end=callback_wrapper,
        generator=generator,
    )
    # highres_output[0].save(f"highres{file_suffix}.png")
    highres_image = decode(sd, highres_latents)[0]
    highres_image = sd.image_processor.postprocess(highres_image, output_type="pil")[0]
    highres_image.save(f"highres{file_suffix}.png")

    # convert final latents and steps back to numpy
    output_latents = interpolate_latents(
        highres_latents,
        size=(input_width, input_height),
        # mode="nearest-exact",
    )
    output_latents = output_latents.cpu().numpy()

    output_steps = np.full(
        (input_width, input_height), profile.steps, dtype=dtype_np_math
    )
    return DiffusionData(
        image=highres_image,
        latents=highres_latents,
        steps=output_steps,
        intermediate_latents=output_latents,
    )


@torch.no_grad()
def splat_texture(sd: Callable, texture_file: str, steps: int = 100):
    logger.info("splatting texture: %s", texture_file)
    device = sd.unet.device

    texture = load_source_texture(texture_file)
    texture = texture / 255.0
    logger.debug(
        "texture stats: shape %s, min %s, max %s",
        texture.shape,
        np.min(texture),
        np.max(texture),
    )

    # encode to latents
    input_image = torch.from_numpy(texture).to(device, dtype=dtype_pt_gpu)
    input_image = input_image.permute(2, 0, 1).unsqueeze(0)
    input_image = sd.image_processor.preprocess(
        input_image, height=TEXTURE_SIZE, width=TEXTURE_SIZE
    )
    latents = sd.vae.encode(input_image)[0]
    latents = latents.cpu().numpy()

    # TODO: apply latents to the world
    # ray_points = on_recast(mesh_vis, scene_geometry, TODO, TODO, TODO, TODO)
    # hit_map = project_voxel_hit_map(ray_points)
    # update_projected_latents(
    #    hit_map, latents, np.full(latents.shape[-2:], steps, dtype=dtype_np_math)
    # )


# endregion


# region: Main Loop
def update_thread(
    checkpoint: str, height: int, width: int, profile: ProfileData, sdxl=False
):
    device, sd = load_sd(checkpoint, sdxl)

    while True:
        update_textures(device, sd, height, width, profile)


def save_numpy_image(image: np.ndarray, filename: str):
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()

    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if np.max(image) <= 1.0:
        image = image * 255.0

    cv2.imwrite(filename, image)


def update_textures(
    material_data: MaterialData,
    # ray_points: np.ndarray,
    # ray_depth: np.ndarray,
    raycast: RaycastData,
    # world_voxels: np.ndarray,
    world: WorldVolume,
    # height: int,
    # width: int,
    profile: ProfileData,
    file_suffix: str = "",
):
    # optimize by looking up the voxel map once and using for all textures
    hit_map = project_voxel_hit_map(raycast)

    # optionally show the projected texture map
    projection = make_projected_texture(hit_map, world, material_data)
    projected_texture = projection.reshape((profile.height, profile.width, 3))
    save_numpy_image(projected_texture, f"projected{file_suffix}.png")

    # optionally show the depth map
    depth_texture = make_projected_depth_texture(raycast)
    save_numpy_image(depth_texture, f"depth{file_suffix}.png")

    return projected_texture, depth_texture


def project_diffusion(
    world: WorldVolume, hit_map: np.ndarray, seed: int | None = None
) -> ProjectionData:
    # select the materials
    projection = make_projected_latents(hit_map, world, seed)
    projection.index = make_projected_index_texture(hit_map, world)

    return projection


# endregion


# region: Main
def main_load_metadata(
    args,
) -> Tuple[MaterialData, ProfileData, np.ndarray, ProfileData]:
    material_data = load_material_data(args.material_data)
    profile_data = load_profile_data(args.profile_data)
    source_texture = load_source_texture(args.source_texture)

    profile = next(p for p in profile_data.profiles if p.name == args.profile)
    return material_data, profile_data, source_texture, profile


def main_create_geometry(
    geometry: GeometryData, extra_geometry: List[o3d.geometry.Geometry3D] = None
) -> Tuple[Any, Any, Any]:
    vertices = o3d.utility.Vector3dVector(geometry.vertices)
    triangles = o3d.utility.Vector3iVector(geometry.triangles)
    mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(geometry.colors)
    mesh.vertex_normals = o3d.utility.Vector3dVector(geometry.normals)
    mesh = mesh.compute_triangle_normals()

    logger.debug("building raycasting scene")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    for extra_mesh in extra_geometry or []:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(extra_mesh))

    # prepare a box for the texture display
    logger.debug("building box for texture display")
    box = o3d.geometry.TriangleMesh.create_box(
        create_uv_map=True, map_texture_to_each_face=True
    )
    box.compute_vertex_normals()
    box.translate([-0.5, -0.5, -2])
    box.triangle_material_ids = o3d.utility.IntVector([1] * len(box.triangles))
    box = o3d.t.geometry.TriangleMesh.from_legacy(box)

    return box, mesh, scene


def main_start_windows(width, height, box, mesh):
    # prepare materials for the point cloud and textures
    logger.debug("loading materials")
    wood_texture = o3d.data.WoodTexture()
    wood_material = o3d.visualization.rendering.MaterialRecord()
    wood_material.shader = "defaultLit"
    wood_material.albedo_img = o3d.io.read_image(wood_texture.albedo_texture_path)

    texture_material = o3d.visualization.rendering.MaterialRecord()
    texture_material.shader = "defaultLit"
    texture_material.albedo_img = o3d.io.read_image(wood_texture.albedo_texture_path)

    diffusion_material = o3d.visualization.rendering.MaterialRecord()
    diffusion_material.shader = "defaultLit"
    diffusion_material.albedo_img = o3d.io.read_image(wood_texture.albedo_texture_path)

    # prepare app windows
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # mesh_vis = o3d.visualization.O3DVisualizer(title="Raycast Diffusion - Mesh", width=width, height=height)
    # mesh_vis.add_geometry({"name": "mesh", "geometry": mesh, "material": texture_material})

    texture_vis = o3d.visualization.O3DVisualizer(
        title="Raycast Diffusion - Texture", width=width, height=height
    )
    texture_vis.add_geometry(
        {"name": "box", "geometry": box, "material": texture_material}
    )

    diffusion_vis = o3d.visualization.O3DVisualizer(
        title="Raycast Diffusion - Diffusion", width=width, height=height
    )
    diffusion_vis.add_geometry(
        {"name": "box", "geometry": box, "material": diffusion_material}
    )

    mesh_vis = o3d.visualization.VisualizerWithKeyCallback()
    mesh_vis.create_window(
        window_name="Raycast Diffusion - Mesh", width=width, height=height
    )
    mesh_vis.add_geometry(mesh)

    return app, mesh_vis, texture_vis, diffusion_vis


def on_recast(camera: CameraData, profile: ProfileData) -> RaycastData:
    logger.info("recasting ray intersections")
    camera.mesh_vis.capture_screen_image("screenshot.png", True)

    # copy view settings
    mesh_view = camera.mesh_vis.get_view_control()
    mesh_pinhole = mesh_view.convert_to_pinhole_camera_parameters()

    # cast rays and create a point cloud
    rays = camera.raycast_scene.create_rays_pinhole(
        intrinsic_matrix=mesh_pinhole.intrinsic.intrinsic_matrix,
        extrinsic_matrix=mesh_pinhole.extrinsic,
        width_px=profile.width,
        height_px=profile.height,
    )
    cast = camera.raycast_scene.cast_rays(rays)

    # build a point cloud from the hit points
    ray_origin = rays[:, :, :3].numpy()
    ray_direction = rays[:, :, 3:].numpy()
    ray_hits = cast["t_hit"].isfinite().numpy()
    ray_hits = np.expand_dims(ray_hits, axis=-1).repeat(3, axis=2)
    ray_length = cast["t_hit"].numpy()
    ray_length = np.expand_dims(ray_length, axis=-1).repeat(3, axis=2)
    tri_normal = cast["primitive_normals"].numpy()

    # find the 90th percentile length and cap the ray depth
    max_depth = np.max(np.where(np.isfinite(ray_length), ray_length, 0))
    ray_depth = np.minimum(ray_length, max_depth)

    # project rays by their length
    ray_points = ray_origin + ray_direction * ray_length
    ray_points = np.floor(ray_points)

    # offset by surface normals for positive faces only
    ray_points -= tri_normal  # np.max(tri_normal, 0)

    # only keep the points that hit a triangle
    ray_points = np.where(
        ray_hits, ray_points, np.asarray(-1, dtype=dtype_np_math)
    ).astype(np.int32)
    return RaycastData(points=ray_points, depth=ray_depth)


def create_view_trajectory(
    front: Tuple[float, float, float],
    lookat: Tuple[float, float, float],
    up: Tuple[float, float, float],
    zoom: float = 1,
):
    return {
        "class_name": "ViewTrajectory",
        "interval": 29,
        "is_loop": False,
        "trajectory": [
            {
                "boundingbox_max": [256.0, 256.0, 120.0],
                "boundingbox_min": [8.0, 0.0, 0.0],
                "field_of_view": 60.0,
                "front": list(front),
                "lookat": list(lookat),
                "up": list(up),
                "zoom": zoom,
            }
        ],
        "version_major": 1,
        "version_minor": 0,
    }


def main():
    args = parse_args()

    height = args.window_height
    width = args.window_width

    logger.info("starting raycast diffusion")
    # load SD first so as not to waste time if it fails
    sd = load_sd(
        args.checkpoint,
        sdxl=args.sdxl,
        decode_only=args.decode_only,
        compiler=args.compiler,
    )

    # load metadata
    material_data, profile_data, source_texture, profile = main_load_metadata(args)

    # make the world voxels
    world_volume = make_world(
        material_data, source_texture, args.ceiling_material, args.floor_material
    )

    # build the world geometry
    geometry_data = make_world_triangles(world_volume, material_data)

    if args.export_geometry:
        logger.info("exporting geometry")
        np.save("world_colors.npy", geometry_data.colors)
        np.save("world_normals.npy", geometry_data.normals)
        np.save("world_triangles.npy", geometry_data.triangles)
        np.save("world_vertices.npy", geometry_data.vertices)
        np.save("world_voxels.npy", world_volume.voxels)

    # load any existing latents or initialize new ones
    load_latents(require_existing=args.decode_only)

    # open3d stuff
    logger.info("converting triangles to mesh")
    box, mesh, scene = main_create_geometry(geometry_data)

    # run the self tests, if selected
    if args.self_test:
        from rcd.tests import run_self_tests

        logger.info("running self tests")
        run_self_tests(args.self_test, sd)
        return

    # start the visualizers
    logger.debug("starting visualizers")
    app, mesh_vis, texture_vis, diffusion_vis = main_start_windows(
        width, height, box, mesh
    )

    diffusion_index = 0

    def on_diffusion(vis):
        nonlocal diffusion_index

        if args.decode_only:
            logger.error("diffusion is not available in preview mode")
            return

        ray_points, ray_depth = on_recast(mesh_vis, scene, width, height)
        logger.warning("running diffusion")
        update_textures(
            material_data,
            ray_points,
            ray_depth,
            height,
            width,
            profile,
            f"-{diffusion_index}",
        )
        index_texture, input_latents, input_steps = project_diffusion(
            world_volume.voxels, ray_points
        )
        run_diffusion(
            sd,
            profile,
            index_texture,
            input_latents,
            input_steps,
            ray_points,
            f"-{diffusion_index}",
        )
        logger.warning("diffusion complete")
        diffusion_index += 1

    def on_preview(vis):
        on_recast(mesh_vis, scene, width, height, world_volume.voxels, material_data)
        logger.warning("running preview")
        preview(sd, profile)
        logger.warning("preview complete")

    def on_splat(vis):
        logger.warning("running splat")
        on_recast(mesh_vis, scene, width, height, world_volume.voxels, material_data)
        splat_texture(sd, args.splat_texture)
        on_preview(vis)
        logger.warning("splat complete")

    mesh_vis.register_key_callback(ord("D"), on_diffusion)
    mesh_vis.register_key_callback(ord("P"), on_preview)
    mesh_vis.register_key_callback(ord("S"), on_splat)
    # mesh_vis.add_action("diffusion", on_diffusion)

    # app.add_window(mesh_vis)
    app.add_window(texture_vis)
    app.add_window(diffusion_vis)

    # diffusion_thread = threading.Thread(target=update_textures, args=(height, width), daemon=True)
    # diffusion_thread.start()

    last_projected_texture = None
    last_diffusion_texture = None
    next_projected_texture = None
    next_diffusion_texture = None

    # pre-render selected angles
    if args.render_angle > 0:
        mesh_controls = mesh_vis.get_view_control()
        for angle in range(0, 360, args.render_angle):
            logger.warning("pre-rendering angle %s", angle)
            mesh_controls.camera_local_rotate(angle, 0)
            ray_points, ray_depth = on_recast(
                mesh_vis, scene, width, height, world_volume.voxels, material_data
            )
            next_projected_texture, next_diffusion_texture = update_textures(
                material_data, ray_points, ray_depth, world_volume.voxels, height, width
            )

    # load camera trajectories from file
    camera_path = {
        "trajectory": [],
    }
    if args.camera_path:
        logger.info("loading camera trajectories from %s", args.camera_path)
        with open(args.camera_path, "r") as f:
            camera_path = load(f, Loader=Loader)

    while True:
        # use the mesh vis window as the indicator to close the app
        if not mesh_vis.poll_events():
            logger.warning("closing")
            app.quit()
            break

        # if there are scripted positions, take screenshots and move to the next one
        camera_trajectory = camera_path.get("trajectory", [])
        if diffusion_index < len(camera_trajectory):
            logger.info("moving to position %s", diffusion_index)
            next_trajectory = camera_trajectory[diffusion_index]
            # hack to create the json structure that o3d is expecting
            trajectory_json = {
                **camera_path,
                "trajectory": [next_trajectory],
            }
            mesh_vis.set_view_status(dumps(trajectory_json))
            mesh_vis.capture_screen_image(f"screenshot-{diffusion_index}.png", True)
            on_diffusion(mesh_vis)
        elif args.exit_after_path:
            logger.warning("camera path complete, closing")
            app.quit()
            break

        # build a texture using the hit voxels
        if (
            next_projected_texture is not None
            and id(next_projected_texture) != last_projected_texture
        ):
            # texture_material.albedo_img = o3d.geometry.Image(next_projected_texture)
            # texture_vis.remove_geometry("box")
            # texture_vis.add_geometry(
            #     {"name": "box", "geometry": box, "material": texture_material}
            # )
            last_projected_texture = id(next_projected_texture)

        # show the diffusion result
        if (
            next_diffusion_texture is not None
            and id(next_diffusion_texture) != last_diffusion_texture
        ):
            # diffusion_material.albedo_img = o3d.geometry.Image(next_diffusion_texture)
            # diffusion_vis.remove_geometry("box")
            # diffusion_vis.add_geometry(
            #     {"name": "box", "geometry": box, "material": diffusion_material}
            # )
            last_diffusion_texture = id(next_diffusion_texture)

        # update the visualizer
        mesh_vis.update_renderer()

    logger.info("cleanup")
    mesh_vis.destroy_window()
    app.quit()
    logger.info("done")


if __name__ == "__main__":
    main()
# endregion
