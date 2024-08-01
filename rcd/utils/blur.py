import torch
import torch.nn.functional as F


def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel using PyTorch."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    grid = torch.stack(torch.meshgrid(coords, coords))
    kernel = torch.exp(-torch.sum(grid**2, dim=0) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def lighten_blur(image, kernel_size=7, sigma=1):
    """Applies a Gaussian blur that only lightens the pixels using PyTorch."""
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Reshape to 4D tensor
    kernel = kernel.to(image.device)

    # Apply convolution
    blurred = F.conv2d(image, kernel, padding=kernel_size // 2)

    # Ensure that the result only lightens the pixels
    lightened_blur = torch.max(image, blurred)

    return lightened_blur
