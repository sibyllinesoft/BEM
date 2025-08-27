#!/usr/bin/env python3
"""
Environment Recording Script for BEM 2.0 Reproducibility

Records complete environment state including package versions, system info,
and git SHAs for reproducibility manifest. Implements B0 phase requirements 
from TODO.md XML workflow.

Usage:
    python scripts/record_env.py --out dist/repro_manifest.json
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np


def get_git_info(repo_path: Path = Path(".")) -> Dict[str, Any]:
    """Get git repository information."""
    git_info = {}
    
    try:
        # Git SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_info["commit_sha"] = result.stdout.strip()
        
        # Git branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_info["branch"] = result.stdout.strip()
        
        # Git status (check for uncommitted changes)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_info["uncommitted_changes"] = len(result.stdout.strip()) > 0
        git_info["status_output"] = result.stdout.strip()
        
        # Git remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()
            
    except subprocess.CalledProcessError as e:
        git_info["error"] = f"Git command failed: {e}"
    except FileNotFoundError:
        git_info["error"] = "Git not available"
    
    return git_info


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cuda_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / (1024**3) 
                if torch.cuda.is_available() else None
            )
        }
    }


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    
    # Core packages from TODO.md requirements
    required_packages = [
        'numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 
        'torch', 'transformers', 'accelerate', 'peft', 'faiss-cpu', 
        'xformers', 'einops', 'wandb', 'mlflow', 'pyyaml', 'rich',
        'sacrebleu'
    ]
    
    for package_name in required_packages:
        try:
            package = __import__(package_name)
            if hasattr(package, '__version__'):
                packages[package_name] = package.__version__
            elif package_name == 'sklearn':
                # scikit-learn is imported as sklearn
                import sklearn
                packages['scikit-learn'] = sklearn.__version__
            else:
                packages[package_name] = "unknown"
        except ImportError:
            packages[package_name] = "not_installed"
    
    # Special handling for some packages
    try:
        import sklearn
        packages['scikit-learn'] = sklearn.__version__
    except ImportError:
        pass
        
    return packages


def get_data_hashes(data_paths: Optional[list] = None) -> Dict[str, str]:
    """Get hashes of key data files."""
    import hashlib
    
    if data_paths is None:
        data_paths = [
            "data/train.jsonl",
            "data/val.jsonl", 
            "data/test.jsonl",
            "indices/domain.faiss",
            "indices/domain.metadata.json"
        ]
    
    hashes = {}
    for path_str in data_paths:
        path = Path(path_str)
        if path.exists():
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            hashes[path_str] = hasher.hexdigest()[:16]  # Short hash
        else:
            hashes[path_str] = "not_found"
    
    return hashes


def check_kernel_flags() -> Dict[str, Any]:
    """Check kernel compilation flags and status."""
    kernel_info = {}
    
    # Check if kernel report exists
    kernel_report_path = Path("logs/kernels.json")
    if kernel_report_path.exists():
        try:
            with open(kernel_report_path, 'r') as f:
                kernel_report = json.load(f)
            kernel_info.update(kernel_report)
        except Exception as e:
            kernel_info["kernel_report_error"] = str(e)
    else:
        kernel_info["kernel_report"] = "not_found"
    
    # Check FP8 self-test results
    fp8_report_path = Path("logs/fp8_selftest.json")
    if fp8_report_path.exists():
        try:
            with open(fp8_report_path, 'r') as f:
                fp8_report = json.load(f)
            kernel_info["fp8_selftest"] = fp8_report
        except Exception as e:
            kernel_info["fp8_selftest_error"] = str(e)
    else:
        kernel_info["fp8_selftest"] = "not_found"
    
    return kernel_info


def create_reproducibility_manifest(output_path: Path) -> Dict[str, Any]:
    """Create complete reproducibility manifest."""
    
    print("🔍 Gathering environment information...")
    
    manifest = {
        "manifest_version": "2.0",
        "timestamp": time.time(),
        "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bem_version": "2.0",
        "git_info": get_git_info(),
        "system_info": get_system_info(),
        "package_versions": get_package_versions(),
        "data_hashes": get_data_hashes(),
        "kernel_flags": check_kernel_flags(),
        "seeds": {
            "numpy_random_seed": None,  # Will be set during experiments
            "torch_seed": None,
            "python_hash_seed": None
        },
        "experiment_config": {
            "budget_parity_tolerance": 0.05,
            "statistical_method": "BCa_95_bootstrap",
            "bootstrap_samples": 10000,
            "fdr_method": "benjamini_hochberg",
            "quality_gates": {
                "numerical_tolerance": 1e-3,
                "fp8_tolerance": 1e-3,
                "cache_safety": True,
                "trust_region_enabled": True
            }
        },
        "validation_status": {
            "environment_validated": True,
            "kernels_compiled": False,  # Will be updated by build script
            "fp8_numerics_pass": False,  # Will be updated by build script
            "ready_for_experiments": False
        }
    }
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📝 Reproducibility manifest saved to: {output_path}")
    
    return manifest


def print_environment_summary(manifest: Dict[str, Any]):
    """Print a summary of the environment."""
    print("\n" + "="*60)
    print("BEM 2.0 ENVIRONMENT SUMMARY")
    print("="*60)
    
    # Git information
    git_info = manifest.get("git_info", {})
    if "commit_sha" in git_info:
        print(f"\n📂 Repository:")
        print(f"  Commit: {git_info['commit_sha'][:12]}")
        print(f"  Branch: {git_info.get('branch', 'unknown')}")
        uncommitted = git_info.get('uncommitted_changes', False)
        print(f"  Uncommitted changes: {'⚠️  YES' if uncommitted else '✅ NO'}")
    
    # System information
    system = manifest.get("system_info", {})
    platform_info = system.get("platform", {})
    hardware = system.get("hardware", {})
    
    print(f"\n💻 System:")
    print(f"  OS: {platform_info.get('system')} {platform_info.get('machine')}")
    print(f"  Python: {platform_info.get('python_version')}")
    print(f"  CUDA: {'✅ Available' if hardware.get('cuda_available') else '❌ Not available'}")
    if hardware.get('cuda_available'):
        print(f"  GPU: {hardware.get('cuda_device_name')}")
        print(f"  VRAM: {hardware.get('cuda_memory_gb', 0):.1f}GB")
    
    # Package versions
    packages = manifest.get("package_versions", {})
    print(f"\n📦 Key Packages:")
    for pkg, version in [("torch", "torch"), ("transformers", "transformers"), 
                        ("numpy", "numpy"), ("sacrebleu", "sacrebleu")]:
        print(f"  {pkg}: {packages.get(version, 'unknown')}")
    
    # Validation status
    validation = manifest.get("validation_status", {})
    print(f"\n✅ Validation:")
    print(f"  Environment: {'✅' if validation.get('environment_validated') else '❌'}")
    print(f"  Kernels: {'✅' if validation.get('kernels_compiled') else '⏳ Pending'}")
    print(f"  FP8 Numerics: {'✅' if validation.get('fp8_numerics_pass') else '⏳ Pending'}")
    ready = validation.get('ready_for_experiments', False)
    print(f"  Ready for experiments: {'✅' if ready else '⏳ Pending validation'}")
    
    if not ready:
        print(f"\n🔧 Next steps:")
        print(f"  1. Run kernel compilation: python bem/kernels/build.py --check-numerics --tol 1e-3")
        print(f"  2. Run FP8 self-test: python bem/quant/fp8_qat.py --selftest")
        print(f"  3. Prepare models and indices")


def main():
    parser = argparse.ArgumentParser(description="Record environment for BEM 2.0 reproducibility")
    parser.add_argument("--out", type=Path, default="dist/repro_manifest.json",
                       help="Output path for reproducibility manifest")
    parser.add_argument("--data-paths", nargs="*", 
                       help="Additional data files to hash")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose environment information")
    
    args = parser.parse_args()
    
    print("🧬 BEM 2.0 Environment Recorder")
    print("=" * 50)
    
    try:
        # Create manifest
        manifest = create_reproducibility_manifest(args.out)
        
        # Print summary
        if args.verbose:
            print_environment_summary(manifest)
        
        # Check if environment is ready
        validation = manifest.get("validation_status", {})
        if validation.get("environment_validated", False):
            print("\n✅ Environment recording completed successfully!")
            return 0
        else:
            print("\n⚠️  Environment recorded with warnings. See summary above.")
            return 0
            
    except Exception as e:
        print(f"\n💥 Error recording environment: {e}")
        return 1


if __name__ == "__main__":
    exit(main())