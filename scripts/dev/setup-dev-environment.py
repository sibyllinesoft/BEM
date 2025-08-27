#!/usr/bin/env python3
"""
Development Environment Setup Script

This script sets up a complete development environment for the BEM project,
including all dependencies, pre-commit hooks, and development tools.

Usage:
    python scripts/dev/setup-dev-environment.py [--gpu] [--research] [--full]
"""

import argparse
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def print_step(message: str) -> None:
    """Print a step message with formatting."""
    print(f"{Colors.BLUE}{Colors.BOLD}==> {message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message with formatting."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message with formatting."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message with formatting."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def run_command(cmd: List[str], description: str, check: bool = True) -> bool:
    """Run a command and handle errors."""
    try:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        print_success(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed: {e}")
        if e.stderr:
            print(f"  Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_step("Checking Python version")
    
    version = sys.version_info
    required_major, required_minor = 3, 9
    
    if version.major != required_major or version.minor < required_minor:
        print_error(f"Python {required_major}.{required_minor}+ required, got {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_system_dependencies() -> bool:
    """Check for required system dependencies."""
    print_step("Checking system dependencies")
    
    dependencies = {
        'git': 'Git version control',
        'make': 'Make build tool',
    }
    
    if platform.system() == 'Linux':
        dependencies.update({
            'build-essential': 'Build tools (install with apt install build-essential)',
            'libpq-dev': 'PostgreSQL development headers (install with apt install libpq-dev)'
        })
    elif platform.system() == 'Darwin':  # macOS
        dependencies.update({
            'clang': 'Clang compiler (install Xcode command line tools)',
        })
    
    missing = []
    for cmd, description in dependencies.items():
        if cmd in ['build-essential', 'libpq-dev']:
            # These are packages, not commands - check differently
            continue
            
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            print_success(f"{description} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(description)
            print_warning(f"{description} not found")
    
    if missing:
        print_warning(f"Missing dependencies: {', '.join(missing)}")
        print("Install missing dependencies and re-run this script.")
        return False
    
    return True


def create_virtual_environment() -> bool:
    """Create and activate virtual environment."""
    print_step("Setting up virtual environment")
    
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print_success("Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command([sys.executable, '-m', 'venv', '.venv'], 
                      "Creating virtual environment"):
        return False
    
    # Get activation script path
    if platform.system() == 'Windows':
        activate_script = venv_path / 'Scripts' / 'activate.bat'
        pip_path = venv_path / 'Scripts' / 'pip.exe'
    else:
        activate_script = venv_path / 'bin' / 'activate'
        pip_path = venv_path / 'bin' / 'pip'
    
    if not pip_path.exists():
        print_error("Virtual environment creation failed - pip not found")
        return False
    
    print_success("Virtual environment created")
    print_warning(f"Activate it with: source {activate_script}")
    return True


def install_dependencies(gpu: bool = False, research: bool = False, full: bool = False) -> bool:
    """Install Python dependencies."""
    print_step("Installing Python dependencies")
    
    # Determine pip command (use virtual environment if available)
    pip_cmd = 'pip'
    if Path('.venv').exists():
        if platform.system() == 'Windows':
            pip_cmd = str(Path('.venv') / 'Scripts' / 'pip.exe')
        else:
            pip_cmd = str(Path('.venv') / 'bin' / 'pip')
    
    # Upgrade pip first
    if not run_command([pip_cmd, 'install', '--upgrade', 'pip', 'setuptools', 'wheel'],
                      "Upgrading pip"):
        return False
    
    # Install base dependencies
    if not run_command([pip_cmd, 'install', '-r', 'requirements.txt'],
                      "Installing base dependencies"):
        return False
    
    if not run_command([pip_cmd, 'install', '-r', 'requirements-dev.txt'],
                      "Installing development dependencies"):
        return False
    
    # Install package in development mode
    if not run_command([pip_cmd, 'install', '-e', '.'],
                      "Installing BEM package in development mode"):
        return False
    
    # Install optional dependencies
    extras = []
    if gpu:
        extras.append('gpu')
    if research:
        extras.append('research')
    if full:
        extras.extend(['gpu', 'research', 'performance', 'docs'])
    
    if extras:
        extra_spec = ','.join(extras)
        if not run_command([pip_cmd, 'install', '-e', f'.[{extra_spec}]'],
                          f"Installing optional dependencies: {extra_spec}"):
            print_warning(f"Failed to install some optional dependencies: {extra_spec}")
    
    return True


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    print_step("Setting up pre-commit hooks")
    
    if not run_command(['pre-commit', '--version'], "Checking pre-commit availability", check=False):
        print_error("pre-commit not available - install development dependencies first")
        return False
    
    if not run_command(['pre-commit', 'install'], "Installing pre-commit hooks"):
        return False
    
    if not run_command(['pre-commit', 'install', '--hook-type', 'commit-msg'],
                      "Installing commit-msg hooks"):
        return False
    
    # Run pre-commit on all files to set up the environment
    print("Running initial pre-commit check (this may take a while)...")
    if not run_command(['pre-commit', 'run', '--all-files'],
                      "Running initial pre-commit check", check=False):
        print_warning("Some pre-commit checks failed - this is normal for initial setup")
    
    return True


def setup_git_hooks() -> bool:
    """Set up additional Git hooks."""
    print_step("Setting up Git hooks")
    
    git_hooks_dir = Path('.git/hooks')
    if not git_hooks_dir.exists():
        print_warning("Git hooks directory not found - are you in a Git repository?")
        return False
    
    # Create pre-push hook for tests
    pre_push_hook = git_hooks_dir / 'pre-push'
    pre_push_content = '''#!/bin/bash
# Pre-push hook to run tests before pushing

echo "Running pre-push checks..."

# Run fast tests only
if ! make test-fast; then
    echo "Tests failed - push aborted"
    exit 1
fi

# Run linting
if ! make lint; then
    echo "Linting failed - push aborted"
    exit 1
fi

echo "Pre-push checks passed"
'''
    
    with open(pre_push_hook, 'w') as f:
        f.write(pre_push_content)
    
    # Make executable
    pre_push_hook.chmod(0o755)
    print_success("Git pre-push hook installed")
    
    return True


def download_models() -> bool:
    """Download required models for development."""
    print_step("Downloading development models")
    
    # Check if download script exists
    download_script = Path('scripts/setup/download_models.py')
    if not download_script.exists():
        print_warning("Model download script not found - skipping model download")
        return True
    
    # Run model download script
    if not run_command([sys.executable, str(download_script), '--setup-dev'],
                      "Downloading development models"):
        print_warning("Model download failed - you can run it later with 'make setup-models'")
        return True
    
    return True


def create_development_config() -> bool:
    """Create development configuration files."""
    print_step("Creating development configuration")
    
    # Create .env.development file
    env_dev_path = Path('.env.development')
    if not env_dev_path.exists():
        env_content = '''# Development environment configuration
# Copy this to .env and modify as needed

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/bem_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed

# Development flags
DEBUG=true
TESTING=false
DEVELOPMENT=true

# Disable external services for development
DISABLE_WANDB=true
DISABLE_MLFLOW=false
DISABLE_SENTRY=true

# Model paths
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data

# Security (development only - change for production)
SECRET_KEY=development-secret-key-change-in-production
DISABLE_AUTH=true
'''
        with open(env_dev_path, 'w') as f:
            f.write(env_content)
        print_success("Created .env.development template")
    
    # Create development docker-compose override
    compose_override = Path('docker-compose.override.yml')
    if not compose_override.exists():
        compose_content = '''version: '3.8'

# Development overrides for docker-compose.yml
# This file is automatically loaded by docker-compose

services:
  bem-app:
    volumes:
      - ./src:/app/src:rw  # Enable live code reloading
      - ./tests:/app/tests:rw
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    ports:
      - "8000:8000"  # Expose API
      - "8888:8888"  # Jupyter notebook
    command: ["uvicorn", "bem_core.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  postgres:
    ports:
      - "5432:5432"  # Expose PostgreSQL
    
  redis:
    ports:
      - "6379:6379"  # Expose Redis
'''
        with open(compose_override, 'w') as f:
            f.write(compose_content)
        print_success("Created docker-compose.override.yml")
    
    return True


def create_vscode_config() -> bool:
    """Create VS Code configuration files."""
    print_step("Creating VS Code configuration")
    
    vscode_dir = Path('.vscode')
    vscode_dir.mkdir(exist_ok=True)
    
    # Settings
    settings_path = vscode_dir / 'settings.json'
    if not settings_path.exists():
        settings = {
            "python.defaultInterpreterPath": "./.venv/bin/python",
            "python.formatting.provider": "black",
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests/"],
            "files.exclude": {
                "**/__pycache__": True,
                "**/.mypy_cache": True,
                "**/.pytest_cache": True,
                "**/logs": True,
                "**/results/outputs": True
            },
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            }
        }
        
        import json
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        print_success("Created VS Code settings")
    
    # Launch configuration
    launch_path = vscode_dir / 'launch.json'
    if not launch_path.exists():
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "Python: Pytest",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["tests/"],
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "BEM: Train Model",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/scripts/train_experiment.py",
                    "args": ["--config", "experiments/S0_baseline.yml"],
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                }
            ]
        }
        
        with open(launch_path, 'w') as f:
            json.dump(launch_config, f, indent=2)
        print_success("Created VS Code launch configuration")
    
    return True


def verify_installation() -> bool:
    """Verify the development environment is working."""
    print_step("Verifying installation")
    
    # Test imports
    try:
        print("Testing package imports...")
        subprocess.run([sys.executable, '-c', 
                       'import bem_core; import bem2; print("✓ Package imports successful")'],
                      check=True)
        print_success("Package imports working")
    except subprocess.CalledProcessError:
        print_error("Package imports failed")
        return False
    
    # Test basic functionality
    try:
        print("Testing basic functionality...")
        test_script = '''
import torch
import bem_core
print(f"✓ BEM version: {bem_core.__version__}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
'''
        subprocess.run([sys.executable, '-c', test_script], check=True)
        print_success("Basic functionality test passed")
    except subprocess.CalledProcessError:
        print_error("Basic functionality test failed")
        return False
    
    # Test development tools
    tools = [
        ('black', 'Code formatter'),
        ('flake8', 'Linter'),
        ('mypy', 'Type checker'),
        ('pytest', 'Test runner'),
        ('pre-commit', 'Pre-commit hooks')
    ]
    
    for tool, description in tools:
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            print_success(f"{description} available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_warning(f"{description} not available")
    
    return True


def print_next_steps() -> None:
    """Print next steps for the user."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Development environment setup complete!{Colors.ENDC}\n")
    
    print("Next steps:")
    print("1. Activate your virtual environment:")
    if platform.system() == 'Windows':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Verify everything works:")
    print("   make test")
    print("   make lint")
    
    print("\n3. Start developing:")
    print("   # Run a simple demo")
    print("   python scripts/demos/demo_simple_bem.py")
    print("   ")
    print("   # Start Jupyter for research")
    print("   make research")
    print("   ")
    print("   # Run the development server")
    print("   docker-compose up")
    
    print("\n4. Useful development commands:")
    print("   make help          # Show all available commands")
    print("   make dev           # Run development checks")
    print("   make clean         # Clean build artifacts")
    print("   make docs          # Build documentation")
    
    print(f"\n{Colors.BLUE}Happy coding! 🚀{Colors.ENDC}")


def main() -> None:
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Set up BEM development environment')
    parser.add_argument('--gpu', action='store_true', 
                       help='Install GPU-specific dependencies')
    parser.add_argument('--research', action='store_true',
                       help='Install research and experimentation tools')
    parser.add_argument('--full', action='store_true',
                       help='Install all optional dependencies')
    parser.add_argument('--skip-models', action='store_true',
                       help='Skip downloading development models')
    parser.add_argument('--skip-vscode', action='store_true',
                       help='Skip VS Code configuration')
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}BEM Development Environment Setup{Colors.ENDC}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}\n")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_system_dependencies():
        sys.exit(1)
    
    # Set up environment
    success = True
    success &= create_virtual_environment()
    success &= install_dependencies(gpu=args.gpu, research=args.research, full=args.full)
    success &= setup_pre_commit()
    success &= setup_git_hooks()
    success &= create_development_config()
    
    if not args.skip_vscode:
        success &= create_vscode_config()
    
    if not args.skip_models:
        success &= download_models()
    
    success &= verify_installation()
    
    if success:
        print_next_steps()
    else:
        print_error("\nSetup completed with some errors. Please review the output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()