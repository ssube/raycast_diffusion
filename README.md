# Raycast Diffusion

> Making Stable Diffusion persistent stable in 3D spaces

This is an effort to make Stable Diffusion output stable in a 3D space, using a persistent latent-space volume to store
partially diffused latents and using them to guide future diffusion paths.

## Setup

This is distributed as a set of Comfy nodes, with an experimental windowed mode and interactive 3D navigation.

To set up the nodes, clone the repository into your ComfyUI `custom_nodes` directory and make sure the requirements have
been installed:

```shell
> cd ~/ComfyUI/custom_nodes

> git clone https://github.com/ssube/raycast_diffusion.git

> cd raycast_diffusion

> pip3 install -r requirements.txt
```

The nodes should appear in the ComfyUI web UI under the `raycast_diffusion` category:

TODO: image

## Workflow

TODO: image

## Methodology

The geometry for the 3D space and projection is extruded from a low-resolution 2D image, where each colors indicates a
material with various parameters such as the image prompt. The prompt embeddings used for the diffusion process are then
assigned to regions of the diffused image using the projected materials, using the tight region masking method from
MultiDiffusion.

The output of the diffusion model's UNet is captured at the final step or an earlier, intermediate step, and stored
in a 3D volume. The latent voxels are projected into screen space for diffusion, then reprojected back into the 3D
space.

The number of diffusion steps that have been run on each voxel are stored along with the output latents. As additional
images are generated, the existing latents are used to guide additional generation using a combination of inpainting
and differential diffusion for soft inpainting with latent guidance.

## Bibliography

- MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation
  - https://multidiffusion.github.io/
- Differential Diffusion: Giving Each Pixel Its Strength
  - https://differential-diffusion.github.io/
- Explaining the SDXL latent space
  - https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
