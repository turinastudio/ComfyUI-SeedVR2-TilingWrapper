# ComfyUI SeedVR2 Tiling Wrapper

A complementary tiling implementation for the official SeedVR2 node package by Numz. This set of nodes is designed to bring advanced, VRAM-aware image tiling, seamless stitching, perceptually accurate color matching, and artifact-free sharpening to your ComfyUI workflows.

## Features & Nodes

### 1. SeedVR2 Tile Splitter (VRAM Aware)
Splits high-resolution images into manageable overlapping tiles to prevent Out-of-Memory (OOM) errors during heavy processing (like upscaling or img2img). 
- **Dynamic VRAM Scaling:** Automatically detects available VRAM (8GB, 12GB, 16GB+) and adjusts the base tile size accordingly, or lets you override it manually.
- **Luminance-based Noise Injection:** Optionally injects noise into tiles to help diffusion models add detail, scaling the noise by luminance to avoid destroying pure black areas.

### 2. SeedVR2 Tile Stitcher (Seamless)
Reconstructs the image from processed tiles without visible seams or grid artifacts.
- **Laplacian Pyramid Blending:** Instead of simple linear alpha blending, this node uses multiresolution pyramid blending. It breaks the tiles into different frequency bands (Laplacian pyramid) and blends them using a Gaussian pyramid mask, ensuring both low-frequency gradients and high-frequency details merge flawlessly.
- **Black Level Fix:** Includes a parameter to optionally shift and correct raised black levels that often occur after complex diffusion processing.

### 3. Advanced Color Match (OKLAB MKL)
A state-of-the-art color matching node designed to transfer the exact color palette from a source image to a target image.
- **OKLAB Color Space:** Operates entirely in the perceptually uniform OKLAB color space, which mimics how the human eye perceives color, preventing the hue shifts common in RGB/HSV matching.
- **Monge-Kantorovich Linear (MKL) Optimal Transport:** Uses statistical Optimal Transport theory to match the covariance matrices of both images. This ensures a mathematical and perceptually accurate transfer of color distributions.
- **Luma Preservation Mode:** Option to only match the 'a' and 'b' chrominance channels, preserving the original lighting and structural depth (L channel) of the target image.

### 4. CAS Luma Sharpening
An advanced sharpening technique that enhances details without artificial halos or color bleeding.
- **YCbCr Luma Isolation:** Converts the image from RGB to YCbCr and exclusively targets the Luma (Y) channel. This ensures that sharpening only affects structure, leaving color channels completely untouched.
- **Contrast Adaptive Sharpening (CAS):** Adapts the sharpening strength based on local contrast to prevent over-sharpening already sharp edges, originally adapted from AMD's FidelityFX.

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/shikasensei-dev/ComfyUI-SeedVR2-TilingWrapper.git
   ```
3. Restart ComfyUI. The nodes will appear under the `SeedVR2_Tiling` category.

## Dependencies

No external dependencies outside of a standard ComfyUI environment are required, as it relies purely on native `torch` and `torch.nn.functional` operations.

## Technologies Used
- PyTorch (for fast, GPU-accelerated tensor operations, border padding, and SVD decomposition)
- Monge-Kantorovich Linear (MKL) Optimal Transport
- Laplacian & Gaussian Image Pyramids
- OKLAB & YCbCr Perceptual Color Spaces
- AMD FidelityFX Contrast Adaptive Sharpening (CAS) algorithm
