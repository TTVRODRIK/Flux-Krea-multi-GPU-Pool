
# **Flux-Krea-multi-GPU-Pool**

A Python-based multi-GPU image generation pipeline using Huggingface Diffusers with LoRA (Low-Rank Adaptation) support. This project distributes image generation workloads across all available GPUs on the system leveraging Python multiprocessing to optimize throughput and speed.

## Features

- Multi-GPU support: Automatically detects GPUs and distributes image generation tasks among them.
- LoRA weight integration for enhanced style adaptation.
- Multiple resolution and style prompt templates.
- Random seed control for reproducibility or random generation.
- Output images saved individually or as a zip archive.
- Configurable generation parameters: number of images, prompt, resolution, guidance scale, and inference steps.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch with CUDA and bfloat16 support
- NVIDIA GPUs with CUDA drivers installed
- Install dependencies with:

```bash
pip install -r requirements.txt
```

Example `requirements.txt` contents:

```
git+https://github.com/huggingface/diffusers.git
git+https://github.com/huggingface/transformers.git
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft
huggingface_hub
sentencepiece
torch
pillow
hf_xet
numpy
torchvision
protobuf
gradio  # optional if using Gradio interface
```

### Usage

Edit the script `app.py` to specify your prompt and generation parameters:

```python
prompt = "Your prompt here"
num_images = 10
```

Run the script:

```bash
python app.py
```

The script will detect your GPUs and perform image generation in parallel, saving images locally and optionally zipping them.

### Function Overview

- `save_image(img)`: Saves PIL image with unique UUID filename.
- `randomize_seed_fn(seed, randomize_seed)`: Handles seed randomization if enabled.
- `apply_style(style_name, positive)`: Applies preset style prompts.
- `generate_on_gpu(args)`: Loads model and LoRA weights on a specified GPU subprocess, generates images for assigned prompt batch.
- `generate(...)`: Main controller function, manages GPU count, divides workload, triggers multiprocessing, and handles output zipping.

### Multi-GPU and Multiprocessing Details

- Utilizes `torch.cuda.set_device(gpu_id)` to direct workload on each subprocess GPU.
- Loads full pipeline and LoRA into each GPU memory context.
- Multiprocessing pool splits image generation workload evenly across all GPUs.
- Each GPU subprocess generates images independently and returns paths and generation duration.
- Zips output images for convenient download if enabled.

### Notes and Recommendations

- Ensure your environment has sufficient GPU VRAM to load the model with LoRA.
- Running on fewer GPUs will automatically fallback to single-GPU generation.
- Multiprocessing incurs some overhead but significantly speeds up batch generation.
- Seed management ensures reproducible or random image generations as needed.
- Modify or extend `style_list` to add custom prompt templates.

## Repository

GitHub: [https://github.com/PRITHIVSAKTHIUR/Flux-Krea-multi-GPU-Pool](https://github.com/PRITHIVSAKTHIUR/Flux-Krea-multi-GPU-Pool)
