import os
import sys
import platform
import psutil
from typing import Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def detect_hardware() -> Dict[str, Any]:
    """
    Detects hardware capabilities including CPU, RAM, GPU, VRAM, and CUDA.
    Returns structured system profile and training recommendations.
    """
    profile = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_processor": platform.processor() or "Unknown CPU",
        "cpu_cores_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_cores_logical": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "torch_available": TORCH_AVAILABLE,
        "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "recommended_device": "cpu",
        "recommended_batch_size": 1,
        "supports_fp16": False,
        "supports_bf16": False,
    }

    if TORCH_AVAILABLE and torch.cuda.is_available():
        profile["cuda_available"] = True
        profile["gpu_count"] = torch.cuda.device_count()
        profile["gpu_name"] = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        profile["vram_total_gb"] = round(vram, 2)
        profile["recommended_device"] = "cuda"
        profile["supports_fp16"] = True
        profile["supports_bf16"] = torch.cuda.is_bf16_supported()
        if vram >= 16.0:
            profile["recommended_batch_size"] = 4
        elif vram >= 8.0:
            profile["recommended_batch_size"] = 2
        else:
            profile["recommended_batch_size"] = 1
    else:
        profile["recommended_device"] = "cpu"
        profile["recommended_batch_size"] = 1
        profile["supports_bf16"] = hasattr(torch, "bfloat16") if TORCH_AVAILABLE else False

    return profile


def print_hardware_report():
    hw = detect_hardware()
    print("=" * 60)
    print(" BHIV TTV HARDWARE & TRAINING ENVIRONMENT REPORT")
    print("=" * 60)
    print(f" OS:                {hw['os']}")
    print(f" Python:            {hw['python_version']}")
    print(f" CPU:               {hw['cpu_processor']} ({hw['cpu_cores_logical']} logical cores)")
    print(f" RAM:               {hw['ram_total_gb']} GB total ({hw['ram_available_gb']} GB available)")
    print(f" PyTorch:           {hw['torch_version']}")
    print(f" CUDA Available:    {hw['cuda_available']}")
    if hw["cuda_available"]:
        print(f" GPU:               {hw['gpu_name']} ({hw['vram_total_gb']} GB VRAM)")
    else:
        print(" GPU:               None (Running on CPU)")
    print("-" * 60)
    print(f" Recommended Device:     {hw['recommended_device']}")
    print(f" Recommended Batch Size: {hw['recommended_batch_size']}")
    print(f" FP16 Mixed Precision:   {hw['supports_fp16']}")
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_report()
