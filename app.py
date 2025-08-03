import torch
from PIL import Image
from diffusers import DiffusionPipeline
import random
import uuid
from typing import Tuple
import numpy as np
import time
import zipfile
import multiprocessing as mp
import os

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max

# Define model and LoRA details
base_model = "black-forest-labs/FLUX.1-Krea-dev"
lora_repo = "strangerzonehf/Flux-Super-Realism-LoRA"
trigger_word = "Super Realism"

# Define styles
style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
DEFAULT_STYLE_NAME = "Style Zero"

def apply_style(style_name: str, positive: str) -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n

def generate_on_gpu(args):
    """Generate images on a specific GPU."""
    prompt, num_images, gpu_id, seed, negative_prompt, use_negative_prompt, style_name, width, height, guidance_scale, num_inference_steps = args

    # Set the specific GPU for this process
    torch.cuda.set_device(gpu_id)
    
    # Load the pipeline and LoRA weights in each process
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora_repo)

    positive_prompt, style_negative_prompt = apply_style(style_name, prompt)

    if use_negative_prompt:
        final_negative_prompt = style_negative_prompt + " " + negative_prompt
    else:
        final_negative_prompt = style_negative_prompt

    final_negative_prompt = final_negative_prompt.strip()

    if trigger_word:
        positive_prompt = f"{trigger_word} {positive_prompt}"

    generator = torch.Generator(device="cuda").manual_seed(seed)

    start_time = time.time()

    images = pipe(
        prompt=positive_prompt,
        negative_prompt=final_negative_prompt if final_negative_prompt else None,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil",
    ).images

    end_time = time.time()
    duration = end_time - start_time

    image_paths = [save_image(img) for img in images]

    return image_paths, duration

def generate(
    prompt: str,
    num_images: int = 1,
    zip_images: bool = True,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    randomize_seed: bool = True,
    style_name: str = DEFAULT_STYLE_NAME,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 28,
):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    if num_gpus > 1:
        # Distribute images across GPUs
        base = num_images // num_gpus
        remainder = num_images % num_gpus
        num_images_list = [base + 1 if i < remainder else base for i in range(num_gpus)]

        # Generate seeds for each GPU
        if randomize_seed:
            main_seed = random.randint(0, MAX_SEED)
            seeds = [main_seed + i for i in range(num_gpus)]
        else:
            seeds = [seed + i for i in range(num_gpus)]

        # Prepare arguments for each GPU process
        args_list = [
            (prompt, num_images_list[i], i, seeds[i], negative_prompt, use_negative_prompt, style_name, width, height, guidance_scale, num_inference_steps)
            for i in range(num_gpus)
        ]

        # Run generation in parallel across GPUs
        with mp.Pool(processes=num_gpus) as p:
            results = p.map(generate_on_gpu, args_list)

        # Collect results
        image_paths = [path for sublist, _ in results for path in sublist]
        durations = [duration for _, duration in results]
        max_duration = max(durations)
        seeds_used = seeds
    else:
        # Single GPU case
        seed = randomize_seed_fn(seed, randomize_seed)
        image_paths, max_duration = generate_on_gpu(
            (prompt, num_images, 0, seed, negative_prompt, use_negative_prompt, style_name, width, height, guidance_scale, num_inference_steps)
        )
        seeds_used = [seed]

    # Zip images if specified
    if zip_images:
        zip_name = str(uuid.uuid4()) + ".zip"
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for i, img_path in enumerate(image_paths):
                zipf.write(img_path, arcname=f"Img_{i}.png")
        print(f"Images zipped in {zip_name}")
    else:
        for path in image_paths:
            print(f"Image saved as {path}")

    # Print generation details
    print(f"Seeds used: {seeds_used}")
    print(f"Generation time: {max_duration:.2f} seconds")

if __name__ == "__main__":
    # Define inputs directly in the script
    prompt = "Super Realism, Headshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style rawHeadshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style rawHeadshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style raw"
    num_images = 10

    # Generate images and zip them
    generate(prompt, num_images=num_images, zip_images=True)
