#!/usr/bin/env python3
"""
BEM Development Environment Setup Script
Sets up complete development environment with all tools and dependencies.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import argparse


class DevEnvironmentSetup:
    """Handles development environment setup for BEM."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.venv_path = self.project_root / ".venv"
        self.requirements_files = {
            "main": "requirements.txt",
            "dev": "requirements-dev.txt",  
            "test": "requirements-test.txt"
        }
        
    def run_command(self, cmd: List[str], check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        print(f"🔧 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode == 0:
                print(f"⚠️  {result.stderr}")
                
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            if not check:
                return e
            raise
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 9:
            print(f"❌ Python {version.major}.{version.minor} detected. Requires Python 3.9+")
            return False
            
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def check_system_dependencies(self) -> bool:
        """Check for system-level dependencies."""
        print("🔍 Checking system dependencies...")
        
        dependencies = {
            "git": "Git version control",
            "curl": "HTTP client for downloads"
        }
        
        all_good = True
        for cmd, description in dependencies.items():
            if shutil.which(cmd) is None:
                print(f"❌ {cmd} not found: {description}")
                all_good = False
            else:
                print(f"✅ {cmd} found")
        
        return all_good
    
    def create_virtual_environment(self, force: bool = False) -> bool:
        """Create Python virtual environment."""
        print("🏗️  Setting up virtual environment...")
        
        if self.venv_path.exists():
            if force:
                print("🗑️  Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)
            else:
                print("✅ Virtual environment already exists")
                return True
        
        try:
            self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
            print(f"✅ Virtual environment created at {self.venv_path}")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    
    def get_pip_command(self) -> List[str]:
        """Get the pip command for the virtual environment."""
        if sys.platform == "win32":
            return [str(self.venv_path / "Scripts" / "pip.exe")]
        else:
            return [str(self.venv_path / "bin" / "pip")]
    
    def install_requirements(self) -> bool:
        """Install Python dependencies."""
        print("📦 Installing Python dependencies...")
        
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        try:
            self.run_command(pip_cmd + ["install", "--upgrade", "pip"])
        except subprocess.CalledProcessError:
            print("⚠️  Failed to upgrade pip, continuing anyway...")
        
        # Install requirements files in order
        for name, filename in self.requirements_files.items():
            req_path = self.project_root / filename
            
            if not req_path.exists():
                if name == "main":
                    print(f"❌ Required file {filename} not found")
                    return False
                else:
                    print(f"⚠️  Optional file {filename} not found, skipping...")
                    continue
            
            print(f"📥 Installing {name} requirements from {filename}...")
            try:
                self.run_command(pip_cmd + ["install", "-r", str(req_path)])
                print(f"✅ {name} requirements installed")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {name} requirements")
                return False
        
        return True
    
    def setup_pre_commit_hooks(self) -> bool:
        """Setup pre-commit hooks for code quality."""
        print("🪝 Setting up pre-commit hooks...")
        
        pip_cmd = self.get_pip_command()
        
        try:
            # Install pre-commit
            self.run_command(pip_cmd + ["install", "pre-commit"])
            
            # Install hooks
            if sys.platform == "win32":
                precommit_cmd = [str(self.venv_path / "Scripts" / "pre-commit.exe")]
            else:
                precommit_cmd = [str(self.venv_path / "bin" / "pre-commit")]
            
            self.run_command(precommit_cmd + ["install"])
            print("✅ Pre-commit hooks installed")
            return True
            
        except subprocess.CalledProcessError:
            print("⚠️  Failed to setup pre-commit hooks, skipping...")
            return True  # Non-critical
    
    def create_dev_requirements(self) -> bool:
        """Create development requirements file if it doesn't exist."""
        dev_req_path = self.project_root / "requirements-dev.txt"
        
        if dev_req_path.exists():
            return True
            
        print("📝 Creating development requirements file...")
        
        dev_requirements = [
            "# Development dependencies",
            "black>=23.0.0",
            "isort>=5.12.0", 
            "mypy>=1.7.0",
            "flake8>=6.0.0",
            "bandit[toml]>=1.7.5",
            "safety>=2.3.0",
            "pre-commit>=3.5.0",
            "",
            "# Testing",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "pytest-timeout>=2.2.0",
            "pytest-mock>=3.12.0",
            "",
            "# Documentation", 
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "mkdocs-gen-files>=0.5.0",
            "",
            "# Build tools",
            "build>=1.0.0",
            "twine>=4.0.0",
            "",
            "# Jupyter (optional)",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ]
        
        with open(dev_req_path, 'w') as f:
            f.write('\n'.join(dev_requirements))
        
        print(f"✅ Created {dev_req_path}")
        return True
    
    def setup_git_hooks(self) -> bool:
        """Setup custom Git hooks."""
        print("📝 Setting up Git hooks...")
        
        hooks_dir = self.project_root / ".git" / "hooks"
        if not hooks_dir.exists():
            print("⚠️  No .git directory found, skipping Git hooks...")
            return True
        
        # Create pre-push hook for running tests
        pre_push_hook = hooks_dir / "pre-push"
        hook_content = '''#!/bin/bash
# Pre-push hook to run tests

echo "🧪 Running tests before push..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run quick tests
python -m pytest tests/unit/ --maxfail=5 -q

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi

echo "✅ Tests passed. Push allowed."
exit 0
'''
        
        with open(pre_push_hook, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        if not sys.platform.startswith('win'):
            pre_push_hook.chmod(0o755)
        
        print("✅ Git hooks installed")
        return True
    
    def create_ide_configs(self) -> bool:
        """Create IDE configuration files."""
        print("⚙️  Creating IDE configuration files...")
        
        # VS Code settings
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings = {
            "python.defaultInterpreterPath": "./.venv/bin/python",
            "python.formatting.provider": "black",
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": False,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests/"],
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                ".pytest_cache": True,
                ".mypy_cache": True,
                "*.egg-info": True
            }
        }
        
        with open(vscode_dir / "settings.json", 'w') as f:
            json.dump(settings, f, indent=2)
        
        # PyCharm/IntelliJ settings would go here
        
        print("✅ IDE configuration files created")
        return True
    
    def verify_setup(self) -> bool:
        """Verify the development environment setup."""
        print("🔍 Verifying development environment...")
        
        pip_cmd = self.get_pip_command()
        
        # Check if key packages are installed
        key_packages = ["torch", "transformers", "black", "pytest"]
        
        try:
            result = self.run_command(pip_cmd + ["list", "--format=json"])
            installed = json.loads(result.stdout)
            installed_names = {pkg["name"].lower() for pkg in installed}
            
            missing = [pkg for pkg in key_packages if pkg.lower() not in installed_names]
            
            if missing:
                print(f"❌ Missing packages: {missing}")
                return False
            
            print("✅ All key packages installed")
            
            # Try importing BEM
            python_cmd = [str(self.venv_path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))]
            
            result = self.run_command(
                python_cmd + ["-c", "import sys; sys.path.insert(0, 'src'); from bem_core import __version__; print(f'BEM version: {__version__}')"],
                check=False
            )
            
            if result.returncode == 0:
                print("✅ BEM package can be imported")
            else:
                print("⚠️  BEM package import issues (expected for initial setup)")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the developer."""
        print("\n" + "="*60)
        print("🎉 Development environment setup complete!")
        print("="*60)
        
        activation_cmd = (
            ".venv\\Scripts\\activate" if sys.platform == "win32" 
            else "source .venv/bin/activate"
        )
        
        print(f"""
📋 Next Steps:

1. Activate virtual environment:
   {activation_cmd}

2. Download required models:
   python scripts/setup/download_models.py --required-only

3. Run validation:
   python scripts/validation/validate_structure.py

4. Run your first demo:
   python scripts/demos/demo_simple_bem.py

5. Run tests:
   pytest tests/ -v

📖 Documentation:
   - Quick Start: docs/QUICK_START.md
   - Developer Guide: docs/guides/DEVELOPER_GUIDE.md
   - Contributing: CONTRIBUTING.md

🔧 Development Tools:
   - Code formatting: black src/ tests/
   - Type checking: mypy src/
   - Testing: pytest tests/ --cov=src
   - Pre-commit hooks: pre-commit run --all-files

Happy coding! 🚀
        """)


def main():
    parser = argparse.ArgumentParser(
        description="Set up BEM development environment"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of virtual environment"
    )
    parser.add_argument(
        "--skip-hooks",
        action="store_true",
        help="Skip Git hooks setup"
    )
    parser.add_argument(
        "--project-root",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    setup = DevEnvironmentSetup(args.project_root)
    
    print("🤖 BEM Development Environment Setup")
    print("="*60)
    
    # Check prerequisites
    if not setup.check_python_version():
        sys.exit(1)
    
    if not setup.check_system_dependencies():
        print("❌ System dependencies missing. Please install them and try again.")
        sys.exit(1)
    
    # Create development requirements
    setup.create_dev_requirements()
    
    # Setup virtual environment
    if not setup.create_virtual_environment(force=args.force):
        sys.exit(1)
    
    # Install dependencies
    if not setup.install_requirements():
        sys.exit(1)
    
    # Setup development tools
    setup.setup_pre_commit_hooks()
    
    if not args.skip_hooks:
        setup.setup_git_hooks()
    
    setup.create_ide_configs()
    
    # Verify setup
    if not setup.verify_setup():
        print("⚠️  Setup completed with warnings")
    
    setup.print_next_steps()


if __name__ == "__main__":
    main()