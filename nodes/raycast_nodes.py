import random
from json import dumps

import folder_paths
import latent_formats
import latent_preview
import numpy as np
import open3d as o3d
import torch

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
    run_img2img,
    run_inpaint,
    update_projected_latents,
    update_textures,
)
from ..rcd.utils.blur import lighten_blur
from ..rcd.utils.latent_correction import (
    center_tensor,
    maximize_tensor,
    soft_clamp_tensor,
)
from .utils import image_to_tensor, pack_latents, tensor_to_image, unpack_latents


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
                "opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "render_color": ("FLOAT3", {}),
                "start_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
                "material_over": ("STRING", {}),
                "material_under": ("STRING", {}),
            },
        }

    # output: Materials
    RETURN_TYPES = ("MATERIALS",)

    FUNCTION = "add_material"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/materials"

    def add_material(
        self,
        materials,
        name,
        prompt,
        color,
        height,
        opacity,
        render_color=None,
        start_height=0,
        material_over=None,
        material_under=None,
    ):
        data = MaterialData(
            name,
            color,
            prompt,
            height,
            opacity,
            render_color or color,
            None,
            start_height,
            material_over,
            material_under,
        )

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
        world = make_world(
            materials, numpy_source, ceiling_material, floor_material, voxels_per_pixel
        )

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
            "optional": {
                "extra_geometry": ("GEOMETRY", {}),
            },
        }

    # output: Camera
    # output: Render
    # output: Depth
    RETURN_TYPES = ("CAMERA", "IMAGE", "IMAGE", "IMAGE")

    FUNCTION = "update_camera"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/geometry"

    def update_camera(
        self, front, lookat, up, zoom, world, materials, profile, extra_geometry=None
    ):
        geometry_data = make_world_triangles(world, materials)
        box, mesh, raycast_scene = main_create_geometry(
            geometry_data, extra_geometry=extra_geometry
        )
        (
            app,
            mesh_vis,
            texture_vis,
            diffusion_vis,
            texture_material,
            diffusion_material,
        ) = main_start_windows(profile.width, profile.height, box, mesh)

        for mesh in extra_geometry or []:
            mesh_vis.add_geometry(mesh)

        trajectory = create_view_trajectory(front, lookat, up, zoom)
        mesh_vis.set_view_status(dumps(trajectory))

        render = mesh_vis.capture_screen_float_buffer(True)
        render = np.asarray(render)

        camera_data = CameraData(mesh_vis, raycast_scene)
        raycast_data = on_recast(camera_data, profile)
        projected, index, depth = update_textures(
            materials, raycast_data, world, profile
        )

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
        diffusion = run_inpaint(
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
        diffusion = run_img2img(
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
                "scheduler": (
                    [
                        "ddim",
                        "deis",
                        "dpm++2m",
                        "dpm++2m-karras",
                        "dpm++2m-sde",
                        "dpm++2m-sde-karras",
                        "dpm++sde",
                        "dpm++sde-karras",
                        "dpm2",
                        "dpm2-karras",
                        "euler",
                        "euler-ancestral",
                        "heun",
                        "lcm",
                        "lms",
                        "lms-karras",
                        "uni-pc",
                    ],
                    {"default": "ddim"},
                ),
            },
        }

    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "load_compile_pipeline"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/pipelines"

    def load_compile_pipeline(
        self,
        checkpoint,
        compiler,
        optimize,
        quantize,
        sdxl,
        tiny_vae,
        xformers,
        scheduler,
    ):
        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        pipeline = load_sd(
            ckpt_path,
            compiler,
            False,
            optimize,
            quantize,
            sdxl,
            tiny_vae,
            xformers,
            scheduler,
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


class LoadModelGeometry:
    # input: Model (STRING)
    # input: Previous Models
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["armadillo", "bunny", "knot", "monkey"], {}),
            },
            "optional": {
                "previous_geometry": ("GEOMETRY", {}),
                "translate": ("FLOAT3", {}),
                "rotate": ("FLOAT3", {}),
                "scale": ("FLOAT", {}),
                "color": ("FLOAT3", {}),
            },
        }

    RETURN_TYPES = ("GEOMETRY",)

    FUNCTION = "add_model_geometry"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/geometry"

    def add_model_geometry(
        self,
        model,
        previous_geometry=None,
        translate=None,
        rotate=None,
        scale=None,
        color=None,
    ):
        previous_geometry = previous_geometry or []

        if model == "armadillo":
            armadillo_mesh = o3d.data.ArmadilloMesh()
            mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
        elif model == "bunny":
            bunny_mesh = o3d.data.BunnyMesh()
            mesh = o3d.io.read_triangle_mesh(bunny_mesh.path)
        elif model == "knot":
            knot_mesh = o3d.data.KnotMesh()
            mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
        elif model == "monkey":
            monkey_mesh = o3d.data.MonkeyModel()
            mesh = o3d.io.read_triangle_mesh(monkey_mesh.path)
        else:
            mesh = o3d.io.read_triangle_mesh(model)

        mesh.compute_vertex_normals()

        if color:
            mesh.paint_uniform_color(color)

        if scale:
            mesh.scale(scale, center=mesh.get_center())

        if rotate:
            mesh.rotate(mesh.get_rotation_matrix_from_xyz(rotate))

        if translate:
            mesh.translate(np.asarray(translate))

        return {
            "result": [[*previous_geometry, mesh]],
            "ui": {"vertices": [len(mesh.vertices)]},
        }


class LightenBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 13, "step": 2}),
                "sigma": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "image-stack": ("IMAGE_STACK", {}),
                "latent": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE_STACK", "LATENT")

    FUNCTION = "lighten_blur"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/filter"

    def lighten_blur(
        self,
        iterations: int,
        kernel_size: int,
        sigma: int,
        image: np.ndarray | None = None,
        image_stack: np.ndarray | None = None,
        latent=None,
    ):
        # convert to tensor
        if image is not None:
            image = image_to_tensor(image)
        elif image_stack is not None:
            image = torch.as_tensor(image_stack)
        elif latent is not None:
            image = unpack_latents(latent)

        # run lighten blur
        for _ in range(iterations):
            image = lighten_blur(image, kernel_size, sigma)

        # convert back to the original type
        if latent is not None:
            result = [None, None, pack_latents(image)]
        elif image_stack is not None:
            result = [None, image.numpy(), None]
        elif image is not None:
            result = [image_to_tensor(image), None, None]

        return {"result": result, "ui": {"image": [image.shape]}}


class LatentColorCorrection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {}),
                "center": ([True, False], {"default": True}),
                "maximize": ([True, False], {"default": True}),
                "soft_clamp": ([True, False], {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "latent_color_correction"
    OUTPUT_NODE = True
    CATEGORY = "raycast_diffusion/filter"

    def latent_color_correction(self, latent, center, maximize, soft_clamp):
        latent = unpack_latents(latent)

        if center:
            latent = center_tensor(latent)

        if maximize:
            latent = maximize_tensor(latent)

        if soft_clamp:
            latent = soft_clamp_tensor(latent)

        latent = pack_latents(latent)
        return {"result": [latent]}
