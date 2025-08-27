#!/usr/bin/env python3
"""
BEM Release Validation Script

This script performs comprehensive validation to ensure the repository is ready for release.
It checks code quality, documentation completeness, security, performance, and more.
"""

import os
import sys
import subprocess
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import importlib.util
import pkg_resources
import re
from packaging import version as pkg_version


@dataclass
class ValidationResult:
    """Represents the result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "ERROR"  # ERROR, WARNING, INFO


class ReleaseValidator:
    """Comprehensive release validation system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load validation configuration."""
        config_path = self.project_root / "scripts" / "validation_config.yml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "coverage_threshold": 85.0,
            "performance_threshold": 2.0,  # seconds
            "memory_limit": 1024,  # MB
            "required_python_versions": ["3.8", "3.9", "3.10", "3.11"],
            "required_files": [
                "README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
                "CODE_OF_CONDUCT.md", "CHANGELOG.md", "requirements.txt",
                "pyproject.toml", "setup.py"
            ],
            "required_directories": [
                "src", "tests", "docs", "scripts", ".github"
            ]
        }
    
    def run_validation(self, comprehensive: bool = False) -> bool:
        """Run all validation checks."""
        print("🔍 Starting BEM Release Validation...")
        print("=" * 60)
        
        # Core validation checks
        self._validate_repository_structure()
        self._validate_code_quality()
        self._validate_documentation()
        self._validate_dependencies()
        self._validate_github_configuration()
        self._validate_security()
        
        if comprehensive:
            self._validate_performance()
            self._validate_installation()
            self._validate_examples()
            self._validate_docker()
            self._validate_licenses()
        
        # Generate report
        return self._generate_report()
    
    def _validate_repository_structure(self):
        """Validate repository file and directory structure."""
        print("\n📁 Validating Repository Structure...")
        
        # Check required files
        for file_name in self.config["required_files"]:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.results.append(ValidationResult(
                    name=f"Required file: {file_name}",
                    passed=True,
                    message=f"{file_name} exists"
                ))
            else:
                self.results.append(ValidationResult(
                    name=f"Required file: {file_name}",
                    passed=False,
                    message=f"{file_name} is missing"
                ))
        
        # Check required directories
        for dir_name in self.config["required_directories"]:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.results.append(ValidationResult(
                    name=f"Required directory: {dir_name}",
                    passed=True,
                    message=f"{dir_name} directory exists"
                ))
            else:
                self.results.append(ValidationResult(
                    name=f"Required directory: {dir_name}",
                    passed=False,
                    message=f"{dir_name} directory is missing"
                ))
        
        # Validate .gitignore
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path) as f:
                gitignore_content = f.read()
                required_patterns = [
                    "__pycache__/", "*.pyc", ".env", ".DS_Store",
                    "dist/", "build/", "*.egg-info/"
                ]
                missing_patterns = [p for p in required_patterns 
                                  if p not in gitignore_content]
                
                if not missing_patterns:
                    self.results.append(ValidationResult(
                        name=".gitignore validation",
                        passed=True,
                        message=".gitignore includes all required patterns"
                    ))
                else:
                    self.results.append(ValidationResult(
                        name=".gitignore validation",
                        passed=False,
                        message=f"Missing patterns: {missing_patterns}",
                        severity="WARNING"
                    ))
    
    def _validate_code_quality(self):
        """Validate code quality metrics."""
        print("\n🔍 Validating Code Quality...")
        
        # Check if linting passes
        try:
            result = subprocess.run(
                ["flake8", "src", "tests", "scripts"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            self.results.append(ValidationResult(
                name="Flake8 linting",
                passed=result.returncode == 0,
                message="All files pass linting" if result.returncode == 0 
                       else f"Linting errors found:\n{result.stdout}"
            ))
        except FileNotFoundError:
            self.results.append(ValidationResult(
                name="Flake8 linting",
                passed=False,
                message="Flake8 not installed",
                severity="WARNING"
            ))
        
        # Check type hints with mypy
        try:
            result = subprocess.run(
                ["mypy", "src"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            self.results.append(ValidationResult(
                name="MyPy type checking",
                passed=result.returncode == 0,
                message="Type checking passed" if result.returncode == 0
                       else f"Type errors found:\n{result.stdout}"
            ))
        except FileNotFoundError:
            self.results.append(ValidationResult(
                name="MyPy type checking",
                passed=False,
                message="MyPy not installed",
                severity="WARNING"
            ))
        
        # Check test coverage
        try:
            result = subprocess.run(
                ["coverage", "run", "-m", "pytest"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                coverage_result = subprocess.run(
                    ["coverage", "report", "--format=json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if coverage_result.returncode == 0:
                    coverage_data = json.loads(coverage_result.stdout)
                    total_coverage = coverage_data["totals"]["percent_covered"]
                    
                    self.results.append(ValidationResult(
                        name="Test coverage",
                        passed=total_coverage >= self.config["coverage_threshold"],
                        message=f"Coverage: {total_coverage:.1f}% (threshold: {self.config['coverage_threshold']}%)",
                        details={"coverage": total_coverage}
                    ))
                else:
                    self.results.append(ValidationResult(
                        name="Test coverage",
                        passed=False,
                        message="Could not generate coverage report"
                    ))
            else:
                self.results.append(ValidationResult(
                    name="Test execution",
                    passed=False,
                    message=f"Tests failed:\n{result.stdout}"
                ))
                
        except FileNotFoundError:
            self.results.append(ValidationResult(
                name="Test coverage",
                passed=False,
                message="Coverage or pytest not installed",
                severity="WARNING"
            ))
    
    def _validate_documentation(self):
        """Validate documentation completeness and quality."""
        print("\n📚 Validating Documentation...")
        
        # Check README.md quality
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            with open(readme_path) as f:
                readme_content = f.read()
                
                required_sections = [
                    "# BEM", "## Installation", "## Quick Start",
                    "## Features", "## License"
                ]
                
                missing_sections = []
                for section in required_sections:
                    if section.lower() not in readme_content.lower():
                        missing_sections.append(section)
                
                self.results.append(ValidationResult(
                    name="README.md completeness",
                    passed=len(missing_sections) == 0,
                    message="README includes all required sections" if not missing_sections
                           else f"Missing sections: {missing_sections}",
                    severity="WARNING" if missing_sections else "INFO"
                ))
        
        # Check API documentation
        docs_path = self.project_root / "docs"
        if docs_path.exists():
            api_doc_path = docs_path / "API.md"
            if api_doc_path.exists():
                self.results.append(ValidationResult(
                    name="API documentation",
                    passed=True,
                    message="API documentation exists"
                ))
            else:
                self.results.append(ValidationResult(
                    name="API documentation",
                    passed=False,
                    message="API.md not found in docs/",
                    severity="WARNING"
                ))
        
        # Check docstring coverage
        try:
            result = subprocess.run(
                ["python", "-c", 
                 "import src; help(src)"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            self.results.append(ValidationResult(
                name="Module documentation",
                passed=True,
                message="Main module is importable and documented",
                severity="INFO"
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Module documentation",
                passed=False,
                message=f"Could not validate module documentation: {e}",
                severity="WARNING"
            ))
    
    def _validate_dependencies(self):
        """Validate dependency management."""
        print("\n📦 Validating Dependencies...")
        
        # Check requirements.txt and pyproject.toml sync
        req_path = self.project_root / "requirements.txt"
        pyproject_path = self.project_root / "pyproject.toml"
        
        if req_path.exists() and pyproject_path.exists():
            with open(req_path) as f:
                req_deps = set(line.split('==')[0].strip() 
                             for line in f if line.strip() and not line.startswith('#'))
            
            try:
                with open(pyproject_path) as f:
                    pyproject_data = yaml.safe_load(f)
                    
                    project_deps = set()
                    if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
                        for dep in pyproject_data['project']['dependencies']:
                            project_deps.add(dep.split('>=')[0].split('==')[0].split('>')[0].strip())
                
                missing_in_pyproject = req_deps - project_deps
                missing_in_requirements = project_deps - req_deps
                
                if not missing_in_pyproject and not missing_in_requirements:
                    self.results.append(ValidationResult(
                        name="Dependency synchronization",
                        passed=True,
                        message="requirements.txt and pyproject.toml are synchronized"
                    ))
                else:
                    self.results.append(ValidationResult(
                        name="Dependency synchronization",
                        passed=False,
                        message=f"Sync issues - Missing in pyproject: {missing_in_pyproject}, Missing in requirements: {missing_in_requirements}",
                        severity="WARNING"
                    ))
                    
            except Exception as e:
                self.results.append(ValidationResult(
                    name="Dependency synchronization",
                    passed=False,
                    message=f"Could not parse pyproject.toml: {e}",
                    severity="WARNING"
                ))
        
        # Check for security vulnerabilities
        try:
            result = subprocess.run(
                ["safety", "check"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            self.results.append(ValidationResult(
                name="Dependency security",
                passed=result.returncode == 0,
                message="No security vulnerabilities found" if result.returncode == 0
                       else f"Security issues found:\n{result.stdout}"
            ))
            
        except FileNotFoundError:
            self.results.append(ValidationResult(
                name="Dependency security",
                passed=False,
                message="Safety tool not installed - cannot check for vulnerabilities",
                severity="WARNING"
            ))
    
    def _validate_github_configuration(self):
        """Validate GitHub repository configuration."""
        print("\n🐙 Validating GitHub Configuration...")
        
        github_dir = self.project_root / ".github"
        
        # Check workflows
        workflows_dir = github_dir / "workflows"
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
            
            required_workflows = ["ci.yml", "release.yml"]
            existing_workflows = [f.name for f in workflow_files]
            
            for required in required_workflows:
                if required in existing_workflows:
                    self.results.append(ValidationResult(
                        name=f"GitHub workflow: {required}",
                        passed=True,
                        message=f"{required} workflow exists"
                    ))
                else:
                    self.results.append(ValidationResult(
                        name=f"GitHub workflow: {required}",
                        passed=False,
                        message=f"{required} workflow missing",
                        severity="WARNING"
                    ))
        
        # Check issue templates
        issue_template_dir = github_dir / "ISSUE_TEMPLATE"
        if issue_template_dir.exists():
            templates = list(issue_template_dir.glob("*.md")) + list(issue_template_dir.glob("*.yml"))
            
            self.results.append(ValidationResult(
                name="Issue templates",
                passed=len(templates) > 0,
                message=f"Found {len(templates)} issue templates" if templates
                       else "No issue templates found",
                severity="INFO" if templates else "WARNING"
            ))
        
        # Check PR template
        pr_template_paths = [
            github_dir / "PULL_REQUEST_TEMPLATE.md",
            github_dir / "pull_request_template.md"
        ]
        
        pr_template_exists = any(p.exists() for p in pr_template_paths)
        self.results.append(ValidationResult(
            name="Pull request template",
            passed=pr_template_exists,
            message="PR template exists" if pr_template_exists else "PR template missing",
            severity="INFO" if pr_template_exists else "WARNING"
        ))
    
    def _validate_security(self):
        """Validate security configuration."""
        print("\n🔒 Validating Security Configuration...")
        
        # Check SECURITY.md exists
        security_path = self.project_root / "SECURITY.md"
        self.results.append(ValidationResult(
            name="Security policy",
            passed=security_path.exists(),
            message="SECURITY.md exists" if security_path.exists() else "SECURITY.md missing"
        ))
        
        # Check for hardcoded secrets (basic check)
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        found_secrets = []
        for python_file in self.project_root.rglob("*.py"):
            try:
                with open(python_file) as f:
                    content = f.read()
                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            found_secrets.extend([(python_file, match) for match in matches])
            except Exception:
                continue
        
        self.results.append(ValidationResult(
            name="Hardcoded secrets check",
            passed=len(found_secrets) == 0,
            message="No hardcoded secrets found" if not found_secrets
                   else f"Potential secrets found: {len(found_secrets)} instances",
            details={"secrets": found_secrets} if found_secrets else None
        ))
    
    def _validate_performance(self):
        """Validate performance benchmarks."""
        print("\n⚡ Validating Performance...")
        
        # Check if benchmark script exists
        benchmark_script = self.project_root / "scripts" / "run_benchmarks.py"
        if benchmark_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(benchmark_script), "--quick"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                self.results.append(ValidationResult(
                    name="Performance benchmarks",
                    passed=result.returncode == 0,
                    message="Benchmarks completed successfully" if result.returncode == 0
                           else f"Benchmarks failed:\n{result.stdout}",
                    details={"output": result.stdout}
                ))
                
            except subprocess.TimeoutExpired:
                self.results.append(ValidationResult(
                    name="Performance benchmarks",
                    passed=False,
                    message="Benchmarks timed out (>60s)",
                    severity="WARNING"
                ))
        else:
            self.results.append(ValidationResult(
                name="Performance benchmarks",
                passed=False,
                message="Benchmark script not found",
                severity="WARNING"
            ))
    
    def _validate_installation(self):
        """Validate installation process."""
        print("\n💾 Validating Installation...")
        
        # Test pip install in temporary environment
        try:
            # Check if package can be built
            result = subprocess.run(
                ["python", "setup.py", "check"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            self.results.append(ValidationResult(
                name="Package build check",
                passed=result.returncode == 0,
                message="Package builds successfully" if result.returncode == 0
                       else f"Package build issues:\n{result.stdout}"
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Package build check",
                passed=False,
                message=f"Could not validate package build: {e}",
                severity="WARNING"
            ))
    
    def _validate_examples(self):
        """Validate example code and tutorials."""
        print("\n📖 Validating Examples...")
        
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            example_files = list(examples_dir.rglob("*.py"))
            
            for example_file in example_files:
                try:
                    # Basic syntax check
                    with open(example_file) as f:
                        compile(f.read(), example_file, 'exec')
                    
                    self.results.append(ValidationResult(
                        name=f"Example: {example_file.name}",
                        passed=True,
                        message="Example syntax is valid",
                        severity="INFO"
                    ))
                    
                except SyntaxError as e:
                    self.results.append(ValidationResult(
                        name=f"Example: {example_file.name}",
                        passed=False,
                        message=f"Syntax error: {e}"
                    ))
        else:
            self.results.append(ValidationResult(
                name="Examples directory",
                passed=False,
                message="No examples directory found",
                severity="WARNING"
            ))
    
    def _validate_docker(self):
        """Validate Docker configuration."""
        print("\n🐳 Validating Docker Configuration...")
        
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path) as f:
                dockerfile_content = f.read()
                
                # Basic Dockerfile checks
                has_from = "FROM" in dockerfile_content
                has_workdir = "WORKDIR" in dockerfile_content
                has_copy = "COPY" in dockerfile_content or "ADD" in dockerfile_content
                
                self.results.append(ValidationResult(
                    name="Dockerfile structure",
                    passed=has_from and has_workdir and has_copy,
                    message="Dockerfile has proper structure" if (has_from and has_workdir and has_copy)
                           else "Dockerfile missing essential instructions"
                ))
        
        compose_path = self.project_root / "docker-compose.yml"
        if compose_path.exists():
            self.results.append(ValidationResult(
                name="Docker Compose",
                passed=True,
                message="docker-compose.yml exists",
                severity="INFO"
            ))
    
    def _validate_licenses(self):
        """Validate license compliance."""
        print("\n⚖️ Validating License Compliance...")
        
        license_path = self.project_root / "LICENSE"
        if license_path.exists():
            with open(license_path) as f:
                license_content = f.read()
                
                # Check for common license types
                license_types = {
                    "MIT": "MIT License",
                    "Apache": "Apache License",
                    "BSD": "BSD License",
                    "GPL": "GNU General Public License"
                }
                
                detected_license = None
                for license_type, identifier in license_types.items():
                    if identifier in license_content:
                        detected_license = license_type
                        break
                
                self.results.append(ValidationResult(
                    name="License detection",
                    passed=detected_license is not None,
                    message=f"Detected {detected_license} license" if detected_license
                           else "Could not detect standard license type",
                    details={"license_type": detected_license}
                ))
        
        # Check NOTICE file for attributions
        notice_path = self.project_root / "NOTICE"
        self.results.append(ValidationResult(
            name="Third-party attributions",
            passed=notice_path.exists(),
            message="NOTICE file exists" if notice_path.exists() else "NOTICE file missing",
            severity="INFO"
        ))
    
    def _generate_report(self) -> bool:
        """Generate and display validation report."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"\n📈 Summary: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {sum(1 for r in self.results if r.severity == 'WARNING')}")
        
        # Group results by severity
        errors = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in self.results if not r.passed and r.severity == "WARNING"]
        
        if errors:
            print("\n🚨 CRITICAL ISSUES (Must Fix Before Release):")
            for result in errors:
                print(f"❌ {result.name}: {result.message}")
        
        if warnings:
            print("\n⚠️  WARNINGS (Should Fix Before Release):")
            for result in warnings:
                print(f"⚠️  {result.name}: {result.message}")
        
        # Success cases
        successes = [r for r in self.results if r.passed]
        if successes:
            print(f"\n✅ PASSED ({len(successes)} checks)")
            for result in successes:
                if result.severity == "INFO":
                    print(f"ℹ️  {result.name}: {result.message}")
        
        # Save detailed report
        report_data = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "success_rate": passed / total * 100
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        report_path = self.project_root / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Return True if no critical errors
        return len(errors) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate BEM repository for release")
    parser.add_argument(
        "--comprehensive", 
        action="store_true",
        help="Run comprehensive validation including performance and examples"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    validator = ReleaseValidator(args.project_root)
    success = validator.run_validation(comprehensive=args.comprehensive)
    
    if success:
        print("\n🎉 VALIDATION PASSED - Repository is ready for release!")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Please fix critical issues before release")
        sys.exit(1)


if __name__ == "__main__":
    main()