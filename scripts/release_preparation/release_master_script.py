#!/usr/bin/env python3
"""
BEM Release Preparation Master Script
Orchestrates the complete release preparation process.
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import tempfile


class ReleasePreparer:
    """Handles complete release preparation for BEM."""
    
    def __init__(self, project_root: Optional[str] = None, dry_run: bool = False):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.dry_run = dry_run
        self.backup_dir = self.project_root / f"release_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Release checklist
        self.checklist = {
            "cleanup": False,
            "documentation": False,
            "git_lfs": False,
            "tests": False,
            "security": False,
            "performance": False,
            "package": False,
            "final_validation": False
        }
    
    def run_command(self, cmd: List[str], check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a shell command."""
        print(f"🔧 Running: {' '.join(cmd)}")
        
        if self.dry_run:
            print("   [DRY RUN - command not executed]")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        
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
    
    def create_backup(self) -> bool:
        """Create backup of current state."""
        print("💾 Creating backup of current state...")
        
        if self.dry_run:
            print("   [DRY RUN - backup not created]")
            return True
        
        try:
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Key files to backup
            backup_files = [
                "README.md",
                "requirements.txt",
                ".gitignore",
                "docs/",
                "experiments/",
                "models/",
                "src/"
            ]
            
            for item in backup_files:
                src_path = self.project_root / item
                if src_path.exists():
                    dst_path = self.backup_dir / item
                    if src_path.is_dir():
                        shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                    else:
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dst_path)
            
            print(f"✅ Backup created at {self.backup_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return False
    
    def run_cleanup(self) -> bool:
        """Run repository cleanup."""
        print("\n🧹 Phase 1: Repository Cleanup")
        print("="*50)
        
        cleanup_script = self.project_root / "scripts/release_preparation/01_cleanup_temporary_files.sh"
        
        if not cleanup_script.exists():
            print("❌ Cleanup script not found")
            return False
        
        try:
            if not self.dry_run:
                self.run_command(["bash", str(cleanup_script)])
            else:
                print("   [DRY RUN - cleanup not executed]")
            
            self.checklist["cleanup"] = True
            print("✅ Repository cleanup completed")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Repository cleanup failed")
            return False
    
    def setup_documentation(self) -> bool:
        """Setup and validate documentation."""
        print("\n📚 Phase 2: Documentation Setup")
        print("="*50)
        
        # Run documentation restructure
        restructure_script = self.project_root / "scripts/release_preparation/02_restructure_documentation.sh"
        if restructure_script.exists():
            try:
                if not self.dry_run:
                    self.run_command(["bash", str(restructure_script)])
                else:
                    print("   [DRY RUN - documentation restructure not executed]")
            except subprocess.CalledProcessError:
                print("⚠️  Documentation restructure had issues, continuing...")
        
        # Generate main README
        readme_script = self.project_root / "scripts/release_preparation/04_create_main_readme.py"
        if readme_script.exists():
            try:
                if not self.dry_run:
                    self.run_command([sys.executable, str(readme_script)])
                else:
                    print("   [DRY RUN - README generation not executed]")
            except subprocess.CalledProcessError:
                print("❌ README generation failed")
                return False
        
        # Validate documentation completeness
        required_docs = [
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "docs/QUICK_START.md",
            "docs/guides/USER_GUIDE.md",
            "docs/guides/DEVELOPER_GUIDE.md"
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not (self.project_root / doc).exists() and not self.dry_run:
                missing_docs.append(doc)
        
        if missing_docs:
            print(f"⚠️  Missing documentation files: {missing_docs}")
            print("   These should be created before release")
        
        self.checklist["documentation"] = True
        print("✅ Documentation setup completed")
        return True
    
    def setup_git_lfs(self) -> bool:
        """Setup Git LFS for large files."""
        print("\n📦 Phase 3: Git LFS Setup")
        print("="*50)
        
        lfs_script = self.project_root / "scripts/release_preparation/03_setup_git_lfs.sh"
        
        if not lfs_script.exists():
            print("❌ Git LFS setup script not found")
            return False
        
        try:
            if not self.dry_run:
                self.run_command(["bash", str(lfs_script)])
            else:
                print("   [DRY RUN - Git LFS setup not executed]")
            
            self.checklist["git_lfs"] = True
            print("✅ Git LFS setup completed")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Git LFS setup failed")
            return False
    
    def run_tests(self) -> bool:
        """Run comprehensive test suite."""
        print("\n🧪 Phase 4: Test Suite")
        print("="*50)
        
        # Run unit tests
        print("Running unit tests...")
        try:
            if not self.dry_run:
                result = self.run_command([
                    sys.executable, "-m", "pytest", 
                    "tests/unit/", 
                    "--cov=src", 
                    "--cov-report=term-missing",
                    "--cov-fail-under=80",
                    "-v"
                ])
            else:
                print("   [DRY RUN - unit tests not executed]")
                
        except subprocess.CalledProcessError:
            print("❌ Unit tests failed")
            return False
        
        # Run validation scripts
        validation_scripts = [
            "scripts/validation/validate_structure.py",
            "scripts/validation/validate_pipeline.py"
        ]
        
        for script in validation_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                try:
                    print(f"Running {script}...")
                    if not self.dry_run:
                        self.run_command([sys.executable, str(script_path)])
                    else:
                        print("   [DRY RUN - validation not executed]")
                except subprocess.CalledProcessError:
                    print(f"⚠️  {script} had issues, continuing...")
        
        self.checklist["tests"] = True
        print("✅ Test suite completed")
        return True
    
    def run_security_scan(self) -> bool:
        """Run security and safety scans."""
        print("\n🛡️ Phase 5: Security Scan")
        print("="*50)
        
        # Install security tools
        try:
            if not self.dry_run:
                self.run_command([sys.executable, "-m", "pip", "install", "bandit[toml]", "safety"])
            else:
                print("   [DRY RUN - security tools not installed]")
        except subprocess.CalledProcessError:
            print("⚠️  Failed to install security tools")
        
        # Run bandit security scan
        print("Running Bandit security scan...")
        try:
            if not self.dry_run:
                self.run_command([
                    sys.executable, "-m", "bandit",
                    "-r", "src/",
                    "-f", "json",
                    "-o", "bandit-report.json"
                ], check=False)
            else:
                print("   [DRY RUN - bandit scan not executed]")
        except subprocess.CalledProcessError:
            print("⚠️  Bandit scan had issues, review manually")
        
        # Run safety dependency check
        print("Running Safety dependency check...")
        try:
            if not self.dry_run:
                self.run_command([
                    sys.executable, "-m", "safety", 
                    "check", 
                    "--json", 
                    "--output", "safety-report.json"
                ], check=False)
            else:
                print("   [DRY RUN - safety check not executed]")
        except subprocess.CalledProcessError:
            print("⚠️  Safety check had issues, review manually")
        
        self.checklist["security"] = True
        print("✅ Security scan completed")
        return True
    
    def run_performance_validation(self) -> bool:
        """Run performance validation."""
        print("\n⚡ Phase 6: Performance Validation")
        print("="*50)
        
        # Check if performance validation script exists
        perf_script = self.project_root / "scripts/validation/validate_performance.py"
        
        if perf_script.exists():
            try:
                print("Running performance validation...")
                if not self.dry_run:
                    self.run_command([sys.executable, str(perf_script), "--quick"])
                else:
                    print("   [DRY RUN - performance validation not executed]")
            except subprocess.CalledProcessError:
                print("⚠️  Performance validation had issues, continuing...")
        else:
            print("⚠️  Performance validation script not found, skipping...")
        
        self.checklist["performance"] = True
        print("✅ Performance validation completed")
        return True
    
    def build_package(self) -> bool:
        """Build Python package."""
        print("\n📦 Phase 7: Package Building")
        print("="*50)
        
        # Install build tools
        try:
            if not self.dry_run:
                self.run_command([sys.executable, "-m", "pip", "install", "build", "twine"])
            else:
                print("   [DRY RUN - build tools not installed]")
        except subprocess.CalledProcessError:
            print("❌ Failed to install build tools")
            return False
        
        # Build package
        try:
            print("Building Python package...")
            if not self.dry_run:
                self.run_command([sys.executable, "-m", "build"])
            else:
                print("   [DRY RUN - package build not executed]")
        except subprocess.CalledProcessError:
            print("❌ Package build failed")
            return False
        
        # Check package
        try:
            print("Checking package integrity...")
            if not self.dry_run:
                self.run_command([sys.executable, "-m", "twine", "check", "dist/*"])
            else:
                print("   [DRY RUN - package check not executed]")
        except subprocess.CalledProcessError:
            print("❌ Package check failed")
            return False
        
        self.checklist["package"] = True
        print("✅ Package building completed")
        return True
    
    def final_validation(self) -> bool:
        """Run final validation checks."""
        print("\n🔍 Phase 8: Final Validation")
        print("="*50)
        
        # Check all required files exist
        required_files = [
            "README.md",
            "LICENSE", 
            "CONTRIBUTING.md",
            "requirements.txt",
            ".gitignore",
            ".gitattributes"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists() and not self.dry_run:
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        # Check Git status
        if not self.dry_run:
            try:
                result = self.run_command(["git", "status", "--porcelain"])
                if result.stdout.strip():
                    print("⚠️  Uncommitted changes detected:")
                    print(result.stdout)
                    print("   Consider committing changes before release")
            except subprocess.CalledProcessError:
                print("⚠️  Could not check Git status")
        
        self.checklist["final_validation"] = True
        print("✅ Final validation completed")
        return True
    
    def generate_release_report(self) -> Dict:
        """Generate release preparation report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "dry_run": self.dry_run,
            "backup_location": str(self.backup_dir) if self.backup_dir.exists() else None,
            "checklist": self.checklist,
            "success": all(self.checklist.values())
        }
        
        # Add file statistics
        if not self.dry_run:
            report["stats"] = {
                "total_files": len(list(self.project_root.rglob("*"))),
                "python_files": len(list(self.project_root.rglob("*.py"))),
                "test_files": len(list((self.project_root / "tests").rglob("*.py"))) if (self.project_root / "tests").exists() else 0,
                "doc_files": len(list((self.project_root / "docs").rglob("*.md"))) if (self.project_root / "docs").exists() else 0
            }
        
        return report
    
    def print_summary(self, report: Dict):
        """Print release preparation summary."""
        print("\n" + "="*60)
        print("🚀 BEM Release Preparation Summary")
        print("="*60)
        
        print(f"📅 Timestamp: {report['timestamp']}")
        print(f"📁 Project Root: {report['project_root']}")
        print(f"🔧 Mode: {'Dry Run' if report['dry_run'] else 'Live Execution'}")
        
        if report.get("backup_location"):
            print(f"💾 Backup: {report['backup_location']}")
        
        print("\n📋 Checklist Status:")
        for task, completed in report["checklist"].items():
            status = "✅" if completed else "❌"
            print(f"   {status} {task.replace('_', ' ').title()}")
        
        if report.get("stats"):
            stats = report["stats"]
            print(f"\n📊 Repository Stats:")
            print(f"   📁 Total files: {stats['total_files']}")
            print(f"   🐍 Python files: {stats['python_files']}")
            print(f"   🧪 Test files: {stats['test_files']}")  
            print(f"   📖 Documentation files: {stats['doc_files']}")
        
        if report["success"]:
            print("\n🎉 Release preparation completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Review generated files and reports")
            print("2. Commit any remaining changes")  
            print("3. Create release tag: git tag -a v2.0.0 -m 'Release v2.0.0'")
            print("4. Push to GitHub: git push origin main --tags")
            print("5. Create GitHub release from pushed tag")
            print("6. Publish package: twine upload dist/*")
        else:
            print("\n⚠️  Release preparation completed with issues!")
            print("Please review the failed items above before proceeding.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BEM repository for release"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Run in dry-run mode without making changes"
    )
    parser.add_argument(
        "--project-root",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip creating backup"
    )
    parser.add_argument(
        "--report-file",
        help="Save report to specified JSON file"
    )
    
    args = parser.parse_args()
    
    preparer = ReleasePreparer(args.project_root, args.dry_run)
    
    print("🚀 BEM Release Preparation")
    print("="*60)
    print(f"Mode: {'🔍 Dry Run' if args.dry_run else '⚡ Live Execution'}")
    print(f"Project: {preparer.project_root}")
    print("="*60)
    
    # Create backup
    if not args.skip_backup:
        if not preparer.create_backup():
            print("❌ Backup creation failed!")
            sys.exit(1)
    
    # Run all phases
    phases = [
        preparer.run_cleanup,
        preparer.setup_documentation,
        preparer.setup_git_lfs,
        preparer.run_tests,
        preparer.run_security_scan,
        preparer.run_performance_validation,
        preparer.build_package,
        preparer.final_validation
    ]
    
    try:
        for phase in phases:
            if not phase():
                print(f"\n❌ Phase failed: {phase.__name__}")
                break
        
        # Generate and save report
        report = preparer.generate_release_report()
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Report saved to {args.report_file}")
        
        preparer.print_summary(report)
        
        if not report["success"]:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Release preparation interrupted!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()