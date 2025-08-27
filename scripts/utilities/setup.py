"""
Setup script for BEM modules with CUDA kernel compilation.

This setup script compiles the high-performance CUDA kernels for BEM operations
and creates a Python package for the BEM system.
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension
import os
import sys

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("Warning: CUDA not available. BEM kernels will not be compiled.")

# CUDA extension configuration
if cuda_available:
    # Get CUDA paths
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is None:
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    
    # CUDA compilation flags
    nvcc_flags = [
        '-O3',                          # Optimization
        '-std=c++17',                   # C++17 standard
        '--use_fast_math',              # Fast math
        '-Xptxas=-v',                   # Verbose PTX assembly
        '-lineinfo',                    # Line info for profiling
        '--expt-relaxed-constexpr',     # Relaxed constexpr
        '--expt-extended-lambda',       # Extended lambda support
        '-gencode', 'arch=compute_80,code=sm_80',  # RTX 3090 Ti (Ampere)
        '-gencode', 'arch=compute_86,code=sm_86',  # Other Ampere cards
        '-gencode', 'arch=compute_75,code=sm_75',  # Turing fallback
        '-DTORCH_API_INCLUDE_EXTENSION_H',
        '-DPYBIND11_COMPILER_TYPE="_gcc"',
        '-DPYBIND11_STDLIB="_libstdcpp"',
        '-DPYBIND11_BUILD_ABI="_cxxabi1011"'
    ]
    
    # C++ compilation flags
    cxx_flags = [
        '-O3',
        '-std=c++17', 
        '-DTORCH_API_INCLUDE_EXTENSION_H',
        '-DWITH_CUDA'
    ]
    
    # Include directories
    include_dirs = [
        pybind11.get_include(),
        torch.utils.cpp_extension.include_paths()[0],  # PyTorch headers
        torch.utils.cpp_extension.include_paths()[1],  # PyTorch CUDA headers
        os.path.join(cuda_home, 'include'),            # CUDA headers
    ]
    
    # Library directories
    library_dirs = [
        os.path.join(cuda_home, 'lib64'),
        torch.utils.cpp_extension.library_paths()[0],
    ]
    
    # Libraries to link
    libraries = ['cudart', 'cublas', 'cusparse', 'curand', 'cufft']
    libraries.extend(torch.utils.cpp_extension.library_paths())
    
    # Define CUDA extension
    fused_bem_ext = cpp_extension.CUDAExtension(
        name='fused_bem_kernels',
        sources=[
            'bem/kernels/fused_generated.cu',
            'bem/kernels/fused_generated.cpp',
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        verbose=True
    )
    
    ext_modules = [fused_bem_ext]
    cmdclass = {'build_ext': cpp_extension.BuildExtension}
    
else:
    # No CUDA - empty extensions
    ext_modules = []
    cmdclass = {}

# Package metadata
setup(
    name='bem_modules',
    version='0.1.0',
    description='High-performance Bolt-on Expert Modules (BEM) with CUDA acceleration',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='BEM Research Team',
    author_email='bem@research.ai',
    url='https://github.com/bem-research/modules',
    
    # Package configuration
    packages=find_packages(),
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'transformers>=4.20.0',
        'accelerate>=0.20.0',
        'datasets>=2.0.0',
        'evaluate>=0.4.0',
        'scikit-learn>=1.0.0',
        'tqdm>=4.64.0',
        'wandb>=0.13.0',
        'pybind11>=2.10.0',
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
        ],
        'benchmark': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'pandas>=1.4.0',
            'jupyterlab>=3.4.0',
        ]
    },
    
    # CUDA extensions
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,  # Required for CUDA extensions
    
    # Package data
    package_data={
        'bem': ['kernels/*.cu', 'kernels/*.cpp', 'kernels/*.h'],
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'bem-profile=scripts.profile_bem:main',
            'bem-validate=scripts.validate_bem:main',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # Keywords
    keywords='deep-learning, transformers, lora, optimization, cuda, pytorch',
)

# Post-installation message
if __name__ == '__main__':
    if cuda_available:
        print("\n" + "="*60)
        print("BEM CUDA Kernels Configuration")
        print("="*60)
        print(f"CUDA Home: {cuda_home}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"Target Architecture: Ampere (RTX 3090 Ti)")
        print(f"Kernel Features: Tensor Cores, Memory Coalescing, Shared Memory")
        print("="*60)
        print("Building high-performance kernels...")
        print("This may take several minutes on first build.")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("BEM Installation - CPU Only")
        print("="*60)
        print("CUDA kernels will not be available.")
        print("Performance will be limited to PyTorch implementation.")
        print("To enable CUDA acceleration, install CUDA toolkit and PyTorch with CUDA.")
        print("="*60 + "\n")