import random
from json import dumps

import folder_paths
import latent_formats
import latent_preview
import numpy as np

from ..rcd.main import (
    BackgroundMaterialData,
    CameraData,
    DiffusionData,
    MaterialData,
    MaterialFile,
    ProfileData,
    ProjectionData,
    RenderData,
    create_view_trajectory,
    load_latents,
    load_sd,
    main_create_geometry,
    main_start_windows,
    make_world,
    make_world_triangles,
    on_recast,
    project_diffusion,
    project_voxel_hit_map,
    run_diffusion,
    run_highres,
    update_projected_latents,
    update_textures,
)
from .utils import image_to_tensor, pack_latents, tensor_to_image, unpack_latents

FLOAT3_LIMIT = 10000.0


class EmptyMaterials:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {}),
                "negative_prompt": ("STRING", {}),
            },
        }

    # output: Materials
    RETURN_TYPES = ("MATERIALS",)

    FUNCTION = "make_materials"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/materials"

    def make_materials(self, prompt, negative_prompt):
        background = BackgroundMaterialData(
            prompt=prompt, negative_prompt=negative_prompt
        )
        materials = MaterialFile(background=background, materials=[])
        return {"result": [materials], "ui": {"materials": []}}


class AddMaterial:
    # input: Materials
    # input: Name
    # input: Prompt
    # input: Color
    # input: Height
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "materials": ("MATERIALS", {}),
                "name": ("STRING", {}),
                "prompt": ("STRING", {}),
                "color": ("FLOAT3", {}),
                "height": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "render_color": ("FLOAT3", {}),
            },
        }

    # output: Materials
    RETURN_TYPES = ("MATERIALS",)

    FUNCTION = "add_material"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/materials"

    def add_material(self, materials, name, prompt, color, height, render_color=None):
        data = MaterialData(name, color, prompt, height, render_color or color)

        # append or replace by name
        replace = False
        for i, m in enumerate(materials.materials):
            if m.name == name:
                materials.materials[i] = data
                replace = True
                break

        if not replace:
            materials.materials.append(data)

        return {
            "result": [materials],
            "ui": {"materials": [m.name for m in materials.materials]},
        }


class CountMaterials:
    # input: Materials
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "materials": ("MATERIALS", {}),
            },
        }

    # output: Count
    RETURN_TYPES = ("INT",)

    FUNCTION = "count_materials"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/materials"

    def count_materials(self, materials):
        return {
            "result": [len(materials.materials)],
            "ui": {"count": [len(materials.materials)]},
        }


class DebugMaterials:
    # input: Materials
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "materials": ("MATERIALS", {}),
            },
        }

    # output: Debug
    RETURN_TYPES = ("STRING",)

    FUNCTION = "debug_materials"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/materials"

    def debug_materials(self, materials):
        names = [m.name for m in materials.materials]
        return {"result": [names], "ui": {"debug": [names]}}


class ExtrudeWorldGeometry:
    # input: Source Texture
    # input: Size
    # input: Materials
    # input: Ceiling Material
    # input: Floor Material
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_texture": ("IMAGE", {}),
                "voxels_per_pixel": (
                    "INT",
                    {"default": 4, "min": 1, "max": 10, "step": 1},
                ),
                "materials": ("MATERIALS", {}),
            },
            "optional": {
                "ceiling_material": ("STRING", {}),
                "floor_material": ("STRING", {}),
            },
        }

    # output: Scene
    RETURN_TYPES = ("WORLD",)

    FUNCTION = "extrude_world"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/geometry"

    def extrude_world(
        self,
        source_texture,
        voxels_per_pixel,
        materials,
        ceiling_material=None,
        floor_material=None,
    ):
        # print("source texture info", source_texture.shape, source_texture.dtype, source_texture.min(), source_texture.max())
        numpy_source = source_texture[0].cpu().numpy() * 255.0
        numpy_source = numpy_source.astype(np.uint8)
        world = make_world(materials, numpy_source, ceiling_material, floor_material)

        return {"result": [world], "ui": {"world_size": world.voxels.shape}}


class LoadWorldLatents:
    # input: World from ExtrudeWorldGeometry
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "world": ("WORLD", {}),
                "always_init": ([True, False], {"default": False}),
                "require_existing": ([True, False], {"default": False}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": (2**31) - 1, "step": 1},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(s, always_init=False, require_existing=False, seed=None, *args):
        print("load world latents", always_init, require_existing, seed)
        if always_init:
            return random.randint(0, 100)
        elif require_existing:
            return "TODO-FILE HASH"
        else:
            return seed

    # output: Latents
    # output: Steps
    RETURN_TYPES = ("WORLD",)

    FUNCTION = "create_latent_volume"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/latents"

    def create_latent_volume(
        self, world, always_init=False, require_existing=False, seed=None
    ):
        print("load world latents", always_init, require_existing, seed)
        loaded_world = load_latents(world, require_existing, always_init, seed)
        return {
            "result": [loaded_world],
            "ui": {
                "latents": [loaded_world.latents.shape],
                "steps": [loaded_world.latents.shape],
            },
        }


class CameraControl:
    # input: Position
    # input: Direction
    # input: Up
    # input: World
    # input: Materials
    # input: Profile
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front": ("FLOAT3", {}),
                "lookat": ("FLOAT3", {}),
                "up": ("FLOAT3", {}),
                "zoom": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "world": ("WORLD", {}),
                "materials": ("MATERIALS", {}),
                "profile": ("PROFILE", {}),
            },
        }

    # output: Camera
    # output: Render
    # output: Depth
    RETURN_TYPES = ("CAMERA", "IMAGE", "IMAGE", "IMAGE")

    FUNCTION = "update_camera"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/geometry"

    def update_camera(self, front, lookat, up, zoom, world, materials, profile):
        geometry_data = make_world_triangles(world, materials)
        box, mesh, raycast_scene = main_create_geometry(geometry_data)
        app, mesh_vis, texture_vis, diffusion_vis = main_start_windows(
            profile.width, profile.height, box, mesh
        )

        trajectory = create_view_trajectory(front, lookat, up, zoom)
        mesh_vis.set_view_status(dumps(trajectory))

        render = mesh_vis.capture_screen_float_buffer(True)
        render = np.asarray(render)

        camera_data = CameraData(mesh_vis, raycast_scene)
        raycast_data = on_recast(camera_data, profile)
        index, depth = update_textures(materials, raycast_data, world, profile)

        return {"result": [camera_data, index, depth, render], "ui": {"camera": []}}


class ProjectWorldLatents:
    # input: World from ExtrudeWorldGeometry or InitWorldLatents
    # input: Camera from CameraControl
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "world": ("WORLD", {}),
                "camera": ("CAMERA", {}),
                "materials": ("MATERIALS", {}),
                "profile": ("PROFILE", {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": (2**31) - 1, "step": 1},
                ),
            },
        }

    # output: Latents
    # output: Steps
    # output: Masks
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE_STACK")

    FUNCTION = "project_scene"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/latents"

    # @classmethod
    # def IS_CHANGED(s, *args):
    #  return random.randint(0, 100)

    def project_scene(self, world, camera, materials, profile, seed=None):
        raycast_data = on_recast(camera, profile)
        hit_map = project_voxel_hit_map(raycast_data)
        projection = project_diffusion(world, hit_map, seed)
        return {
            "result": [projection.latents, projection.steps, projection.index],
            "ui": {
                "latents": [projection.latents.shape],
                "steps": [projection.steps.shape],
                "masks": [projection.index.shape],
            },
        }


class ComfyPipelineModelWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def latent_format(self):
        return latent_formats.SDXL()


class ComfyPipelineWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def load_device(self):
        return self.pipeline.unet.device

    @property
    def model(self):
        return ComfyPipelineModelWrapper(self.pipeline)


class MultiDiffControlInpaint:
    # input: Pipeline
    # input: Latents
    # input: Depth
    # input: Masks
    # input: Steps
    # input: Materials
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", {}),
                "latents": ("LATENT", {}),
                "steps": ("IMAGE", {}),
                "masks": ("IMAGE_STACK", {}),
                "depth": ("IMAGE", {}),
                "materials": ("MATERIALS", {}),
                "profile": ("PROFILE", {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": (2**31) - 1, "step": 1},
                ),
                "color_correction": ([True, False], {"default": False}),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "controlnet_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    # output: Image
    # output: Final Latents
    # output: Intermediate Latents
    # output: Steps
    RETURN_TYPES = ("IMAGE", "LATENT", "LATENT", "IMAGE")

    FUNCTION = "run_diffusion"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/pipelines"

    def run_diffusion(
        self,
        pipeline,
        latents,
        depth,
        masks,
        steps,
        materials,
        profile,
        seed,
        strength=1.0,
        controlnet_strength=0.6,
        color_correction=False,
    ):
        comfy_model = ComfyPipelineWrapper(pipeline)
        callback = latent_preview.prepare_callback(comfy_model, profile.steps)

        render = RenderData(index=masks, depth=depth)
        projection = ProjectionData(index=masks, latents=latents, steps=steps)
        diffusion = run_diffusion(
            pipeline,
            profile,
            render,
            projection,
            materials,
            callback=callback,
            strength=strength,
            controlnet_strength=controlnet_strength,
            seed=seed,
            color_correction=color_correction,
        )

        images = image_to_tensor(diffusion.image)
        samples = pack_latents(diffusion.latents)
        intermediate_samples = pack_latents(diffusion.intermediate_latents)

        return {
            "result": [images, samples, intermediate_samples, diffusion.steps],
            "ui": {
                "image": [diffusion.image.size],
                "steps": [diffusion.steps.shape],
                "latents": [diffusion.latents.shape],
            },
        }


class MultiControlImg2Img:
    # input: Pipeline
    # input: Latents from ProjectScene
    # input: Masks from ProjectScene
    # TODO: input: steps
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", {}),
                "latents": ("LATENT", {}),
                "steps": ("IMAGE", {}),
                "masks": ("IMAGE_STACK", {}),
                "depth": ("IMAGE", {}),
                "materials": ("MATERIALS", {}),
                "profile": ("PROFILE", {}),
                "previous_image": ("IMAGE", {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": (2**31) - 1, "step": 1},
                ),
                "color_correction": ([True, False], {"default": False}),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "controlnet_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "LATENT", "IMAGE")

    FUNCTION = "run_diffusion"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/pipelines"

    # output: Latents
    def run_diffusion(
        self,
        pipeline,
        latents,
        steps,
        masks,
        depth,
        profile,
        previous_image,
        materials,
        seed,
        strength=0.4,
        controlnet_strength=0.6,
        color_correction=False,
    ):
        if isinstance(latents, dict) and "samples" in latents:
            latents = latents["samples"]

        comfy_model = ComfyPipelineWrapper(pipeline)
        total_steps = int((profile.steps * 2) * strength)
        print(f"preparing callback for {total_steps} total steps")
        callback = latent_preview.prepare_callback(comfy_model, total_steps)

        # convert previous_image back to PIL
        input_image = tensor_to_image(previous_image)

        render = RenderData(index=masks, depth=depth)
        projection = ProjectionData(index=masks, latents=latents, steps=steps)
        diffusion = run_highres(
            pipeline,
            profile,
            render,
            projection,
            input_image,
            materials,
            callback=callback,
            strength=strength,
            controlnet_strength=controlnet_strength,
            seed=seed,
            color_correction=color_correction,
        )

        images = image_to_tensor(diffusion.image)
        samples = pack_latents(diffusion.latents)
        intermediate_samples = pack_latents(diffusion.intermediate_latents)

        return {
            "result": [images, samples, intermediate_samples, diffusion.steps],
            "ui": {
                "image": [diffusion.image.size],
                "steps": [diffusion.steps.shape],
                "latents": [diffusion.latents.shape],
            },
        }


class UpdateWorldLatents:
    # input: World from ExtrudeWorldGeometry
    # input: Camera from CameraControl
    # input: Latents from MultiDiffControlInpaint
    # input: Steps from MultiDiffControlInpaint
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "world": ("WORLD", {}),
                "camera": ("CAMERA", {}),
                "latents": ("LATENT", {}),
                "steps": ("IMAGE", {}),
                "profile": ("PROFILE", {}),
            },
        }

    # output: World
    # output: Updated voxel count
    RETURN_TYPES = ("WORLD", "INT")

    FUNCTION = "update_world"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/latents"

    def update_world(self, world, camera, latents, steps, profile):
        latents = unpack_latents(latents)
        latents = latents.cpu().numpy()

        raycast_data = on_recast(camera, profile)
        hit_map = project_voxel_hit_map(raycast_data)
        diffusion = DiffusionData(
            image=None, latents=latents, steps=steps, intermediate_latents=latents
        )
        updated = update_projected_latents(hit_map, world, diffusion)

        return {"result": [world, updated], "ui": {"updated_voxels": [updated]}}


class LoadCompilePipeline:
    # input: Checkpoint
    # input: Compiler
    # input: Optimize
    # input: Quantize
    # input: SDXL
    # input: Tiny VAE
    # input: Xformers
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"), {}),
                "compiler": (["none", "torch", "sfast", "onediff"], {}),
                "optimize": ([True, False], {}),
                "quantize": ([False, True], {}),
                "sdxl": ([True, False], {}),
                "tiny_vae": ([False, True], {}),
                "xformers": ([False, True], {}),
            },
        }

    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "load_compile_pipeline"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/pipelines"

    def load_compile_pipeline(
        self, checkpoint, compiler, optimize, quantize, sdxl, tiny_vae, xformers
    ):
        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        pipeline = load_sd(
            ckpt_path, compiler, False, optimize, quantize, sdxl, tiny_vae, xformers
        )
        return {"result": [pipeline], "ui": {"pipeline": []}}


class MakeProfile:
    # input: Name
    # input: Bootstrapping Steps
    # input: CFG Scale
    # input: Steps
    # input: Width
    # input: Height
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {}),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 50.0, "step": 0.01},
                ),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "bootstrapping_steps": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100, "step": 1},
                ),
                "capture_steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 100, "step": 1},
                ),
                "width": ("INT", {"default": 512, "min": 1, "max": 1024, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 1024, "step": 1}),
                # TODO: add start_at
                # TODO: add stop_at
            },
        }

    RETURN_TYPES = ("PROFILE",)

    FUNCTION = "make_profile"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/pipelines"

    def make_profile(
        self, name, cfg_scale, steps, bootstrapping_steps, capture_steps, width, height
    ):
        profile = ProfileData(
            name, cfg_scale, steps, bootstrapping_steps, capture_steps, width, height
        )
        return {"result": [profile], "ui": {"profile": [profile.name]}}


class SaveWorldLatents:
    # input: World from ExtrudeWorldGeometry
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "world": ("WORLD", {}),
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "save_world_latents"
    OUTPUT_NODE = False
    CATEGORY = "raycast_diffusion/latents"

    def save_world_latents(self, world):
        # TODO: save the world latents
        return {"result": [world], "ui": {"world": [world.voxels.shape]}}