# GPU deployment

CPU development uses the OpenCV video adapter and viseme lip-sync fallback. It is suitable for API/UI integration tests, not high-quality diffusion video production.

For Wan, deploy a Linux/NVIDIA CUDA worker with adequate disk for model weights and generated clips, install a CUDA-compatible PyTorch build plus Diffusers/Transformers/Accelerate, set `VIDEO_PROVIDER=wan`, `WAN_DEVICE=cuda`, and an appropriate `WAN_DTYPE`. Plan roughly 24 GB+ VRAM for 720p experimentation; capacity depends on frames, batch size, offloading, and model release. Confirm CUDA/GPU/VRAM at worker startup and route large jobs to a queue-backed GPU worker.
