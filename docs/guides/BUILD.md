# Build Environment Documentation

## System Configuration
- **Date**: 2025-08-22
- **OS**: Linux
- **Platform**: x86_64
- **GPU**: NVIDIA GeForce RTX 3090 Ti (24GB VRAM)
- **CUDA**: 12.9
- **Driver**: 575.64.03

## Python Environment
- **Python Version**: 3.13.3
- **Virtual Environment**: .venv/
- **Package Manager**: pip

## Key Dependencies
- PyTorch: 2.8.0 (with CUDA 12.1 support)
- Transformers: 4.55.4
- CUDA Runtime: Verified working

## Installation Commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft faiss-cpu sentencepiece datasets xformers
pip install wandb mlflow rich numpy scipy scikit-learn einops fire
```

## Verification
- CUDA availability: ✓ Verified
- GPU device detected: ✓ NVIDIA GeForce RTX 3090 Ti
- Package dependencies: ✓ All installed successfully

## Repository Structure
```
/
├─ bem/                    # Core BEM implementation
│   ├─ kernels/           # Fused CUDA kernels
│   └─ shim/              # Per-family layer maps
├─ manifests/             # Configuration files
├─ indices/               # FAISS retrieval indices
├─ experiments/           # Training configurations
├─ logs/                  # Experiment logs
├─ eval/                  # Evaluation harnesses
├─ requirements.txt       # Python dependencies
└─ BUILD.md              # This file
```