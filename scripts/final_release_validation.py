#!/usr/bin/env python3
"""
Final Release Validation Script for BEM

This script performs the ultimate validation check before GitHub release,
ensuring every aspect of the repository is polished and ready for public release.
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import re
from datetime import datetime
import tempfile
import shutil


@dataclass
class ValidationCheck:
    """Represents a single validation check."""
    name: str
    category: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None


class FinalReleaseValidator:
    """Comprehensive final release validation system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.checks: List[ValidationCheck] = []
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load validation configuration."""
        config_path = self.project_root / "scripts" / "validation_config.yml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def run_final_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Run comprehensive final validation."""
        print("🚀 Final BEM Release Validation")
        print("=" * 60)
        print("This validation ensures 100% readiness for GitHub release.")
        print()
        
        # Run all validation categories
        self._validate_repository_completeness()
        self._validate_code_quality_standards()
        self._validate_documentation_excellence()
        self._validate_security_hardening()
        self._validate_performance_benchmarks()
        self._validate_ci_cd_workflows()
        self._validate_community_readiness()
        self._validate_marketing_assets()
        self._validate_legal_compliance()
        self._validate_release_artifacts()
        
        return self._generate_final_report()
    
    def _validate_repository_completeness(self):
        """Validate repository structure and essential files."""
        print("📁 Validating Repository Completeness...")
        
        essential_files = [
            "README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
            "CODE_OF_CONDUCT.md", "CHANGELOG.md", "RELEASE_CHECKLIST.md",
            "requirements.txt", "pyproject.toml", "setup.py", "MANIFEST.in",
            "Makefile", "Dockerfile", "docker-compose.yml", ".gitignore",
            ".github/workflows/ci.yml", ".github/workflows/release.yml",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/PULL_REQUEST_TEMPLATE.md"
        ]
        
        for file_path in essential_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                # Check file is not empty
                if full_path.stat().st_size > 0:
                    self.checks.append(ValidationCheck(
                        name=f"File: {file_path}",
                        category="Repository Structure",
                        passed=True,
                        message=f"✅ {file_path} exists and is not empty"
                    ))
                else:
                    self.checks.append(ValidationCheck(
                        name=f"File: {file_path}",
                        category="Repository Structure",
                        passed=False,
                        message=f"❌ {file_path} exists but is empty",
                        fix_suggestion=f"Add content to {file_path}"
                    ))
            else:
                self.checks.append(ValidationCheck(
                    name=f"File: {file_path}",
                    category="Repository Structure",
                    passed=False,
                    message=f"❌ {file_path} is missing",
                    fix_suggestion=f"Create {file_path} following project templates"
                ))
        
        # Validate directory structure
        required_dirs = ["src", "tests", "docs", "scripts", "examples"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                # Check directory has content
                content_count = len(list(dir_path.rglob("*")))
                if content_count > 0:
                    self.checks.append(ValidationCheck(
                        name=f"Directory: {dir_name}",
                        category="Repository Structure", 
                        passed=True,
                        message=f"✅ {dir_name}/ exists with {content_count} items"
                    ))
                else:
                    self.checks.append(ValidationCheck(
                        name=f"Directory: {dir_name}",
                        category="Repository Structure",
                        passed=False,
                        message=f"❌ {dir_name}/ exists but is empty",
                        severity="WARNING"
                    ))
            else:
                self.checks.append(ValidationCheck(
                    name=f"Directory: {dir_name}",
                    category="Repository Structure",
                    passed=False,
                    message=f"❌ {dir_name}/ directory missing"
                ))
    
    def _validate_code_quality_standards(self):
        """Validate code meets quality standards."""
        print("🔍 Validating Code Quality Standards...")
        
        # Check linting
        for tool, command in [
            ("flake8", ["flake8", "src", "tests", "scripts"]),
            ("mypy", ["mypy", "src"]),
            ("black", ["black", "--check", "src", "tests", "scripts"]),
            ("isort", ["isort", "--check-only", "src", "tests", "scripts"])
        ]:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                self.checks.append(ValidationCheck(
                    name=f"Code Quality: {tool}",
                    category="Code Quality",
                    passed=result.returncode == 0,
                    message=f"✅ {tool} passed" if result.returncode == 0 
                           else f"❌ {tool} failed: {result.stdout[:200]}",
                    details={"stdout": result.stdout, "stderr": result.stderr}
                ))
                
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                self.checks.append(ValidationCheck(
                    name=f"Code Quality: {tool}",
                    category="Code Quality",
                    passed=False,
                    message=f"❌ Could not run {tool}: {str(e)}",
                    severity="WARNING"
                ))
        
        # Check test coverage
        try:
            # Run tests with coverage
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Check coverage report
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        total_coverage = coverage_data["totals"]["percent_covered"]
                        
                        threshold = self.config.get("coverage_threshold", 85.0)
                        self.checks.append(ValidationCheck(
                            name="Test Coverage",
                            category="Code Quality",
                            passed=total_coverage >= threshold,
                            message=f"{'✅' if total_coverage >= threshold else '❌'} Coverage: {total_coverage:.1f}% (threshold: {threshold}%)",
                            details={"coverage": total_coverage, "threshold": threshold}
                        ))
                else:
                    self.checks.append(ValidationCheck(
                        name="Test Coverage",
                        category="Code Quality",
                        passed=False,
                        message="❌ Coverage report not generated"
                    ))
            else:
                self.checks.append(ValidationCheck(
                    name="Test Execution",
                    category="Code Quality", 
                    passed=False,
                    message=f"❌ Tests failed: {result.stdout[:200]}"
                ))
                
        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Test Coverage",
                category="Code Quality",
                passed=False,
                message=f"❌ Could not run coverage: {str(e)}",
                severity="WARNING"
            ))
    
    def _validate_documentation_excellence(self):
        """Validate documentation quality and completeness."""
        print("📚 Validating Documentation Excellence...")
        
        # Check README quality
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            with open(readme_path) as f:
                readme_content = f.read()
                
                required_sections = [
                    "# BEM", "## Features", "## Installation", "## Quick Start",
                    "## Documentation", "## Contributing", "## License",
                    "## Citation", "## Acknowledgments"
                ]
                
                missing_sections = []
                for section in required_sections:
                    if not re.search(section.replace("##", r"##?\s*"), readme_content, re.IGNORECASE):
                        missing_sections.append(section)
                
                self.checks.append(ValidationCheck(
                    name="README Completeness",
                    category="Documentation",
                    passed=len(missing_sections) == 0,
                    message=f"{'✅' if not missing_sections else '❌'} README sections {'complete' if not missing_sections else f'missing: {missing_sections}'}",
                    details={"missing_sections": missing_sections}
                ))
                
                # Check README length (should be substantial)
                word_count = len(readme_content.split())
                min_words = 500
                self.checks.append(ValidationCheck(
                    name="README Depth",
                    category="Documentation",
                    passed=word_count >= min_words,
                    message=f"{'✅' if word_count >= min_words else '❌'} README has {word_count} words (min: {min_words})",
                    details={"word_count": word_count, "min_words": min_words}
                ))
        
        # Check documentation directories
        docs_checks = [
            ("docs/API.md", "API documentation"),
            ("docs/QUICK_START.md", "Quick start guide"),
            ("docs/USER_GUIDE.md", "User guide"),
            ("docs/DEVELOPER_GUIDE.md", "Developer guide"),
            ("docs/TROUBLESHOOTING.md", "Troubleshooting guide"),
            ("examples/", "Examples directory")
        ]
        
        for doc_path, description in docs_checks:
            full_path = self.project_root / doc_path
            if full_path.exists():
                if full_path.is_file():
                    size = full_path.stat().st_size
                    self.checks.append(ValidationCheck(
                        name=f"Documentation: {description}",
                        category="Documentation",
                        passed=size > 500,  # At least 500 bytes
                        message=f"{'✅' if size > 500 else '❌'} {description} {'exists with substantial content' if size > 500 else 'exists but may be too short'}",
                        details={"size": size}
                    ))
                elif full_path.is_dir():
                    content_count = len(list(full_path.rglob("*")))
                    self.checks.append(ValidationCheck(
                        name=f"Documentation: {description}",
                        category="Documentation",
                        passed=content_count > 0,
                        message=f"{'✅' if content_count > 0 else '❌'} {description} {'has content' if content_count > 0 else 'is empty'}",
                        details={"content_count": content_count}
                    ))
            else:
                self.checks.append(ValidationCheck(
                    name=f"Documentation: {description}",
                    category="Documentation",
                    passed=False,
                    message=f"❌ {description} is missing",
                    severity="WARNING"
                ))
    
    def _validate_security_hardening(self):
        """Validate security measures are in place."""
        print("🔒 Validating Security Hardening...")
        
        # Check security policy
        security_path = self.project_root / "SECURITY.md"
        if security_path.exists():
            with open(security_path) as f:
                security_content = f.read()
                required_sections = [
                    "Reporting", "Supported Versions", "Security Policy"
                ]
                
                has_all_sections = all(
                    section.lower() in security_content.lower() 
                    for section in required_sections
                )
                
                self.checks.append(ValidationCheck(
                    name="Security Policy",
                    category="Security",
                    passed=has_all_sections,
                    message=f"{'✅' if has_all_sections else '❌'} Security policy {'complete' if has_all_sections else 'incomplete'}"
                ))
        else:
            self.checks.append(ValidationCheck(
                name="Security Policy",
                category="Security",
                passed=False,
                message="❌ SECURITY.md missing"
            ))
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'(?i)(password|pwd|pass)\s*[=:]\s*["\'][^"\']+["\']', "password"),
            (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][^"\']+["\']', "API key"),
            (r'(?i)(secret|secret[_-]?key)\s*[=:]\s*["\'][^"\']+["\']', "secret"),
            (r'(?i)(token|access[_-]?token)\s*[=:]\s*["\'][^"\']+["\']', "token"),
            (r'-----BEGIN [A-Z ]+ PRIVATE KEY-----', "private key"),
            (r'(?i)mongodb://[^\\s/]+:[^\\s/]+@', "MongoDB connection string"),
            (r'(?i)mysql://[^\\s/]+:[^\\s/]+@', "MySQL connection string")
        ]
        
        found_secrets = []
        for python_file in self.project_root.rglob("*.py"):
            if "/.git/" in str(python_file) or "/venv/" in str(python_file):
                continue
                
            try:
                with open(python_file) as f:
                    content = f.read()
                    for pattern, secret_type in secret_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            found_secrets.append((python_file, secret_type, len(matches)))
            except Exception:
                continue
        
        self.checks.append(ValidationCheck(
            name="Hardcoded Secrets Check",
            category="Security",
            passed=len(found_secrets) == 0,
            message=f"{'✅' if not found_secrets else '❌'} {'No hardcoded secrets found' if not found_secrets else f'Found {len(found_secrets)} potential secrets'}",
            details={"found_secrets": found_secrets}
        ))
        
        # Check security workflows
        security_workflow = self.project_root / ".github" / "workflows" / "security.yml"
        self.checks.append(ValidationCheck(
            name="Security Workflow",
            category="Security",
            passed=security_workflow.exists(),
            message=f"{'✅' if security_workflow.exists() else '❌'} Security workflow {'exists' if security_workflow.exists() else 'missing'}"
        ))
    
    def _validate_performance_benchmarks(self):
        """Validate performance benchmarking setup."""
        print("⚡ Validating Performance Benchmarks...")
        
        # Check benchmark script exists
        benchmark_script = self.project_root / "scripts" / "run_benchmarks.py"
        if benchmark_script.exists():
            self.checks.append(ValidationCheck(
                name="Benchmark Script",
                category="Performance",
                passed=True,
                message="✅ Benchmark script exists"
            ))
            
            # Try to run quick benchmark
            try:
                result = subprocess.run(
                    ["python", str(benchmark_script), "--quick"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                self.checks.append(ValidationCheck(
                    name="Benchmark Execution",
                    category="Performance",
                    passed=result.returncode == 0,
                    message=f"{'✅' if result.returncode == 0 else '❌'} Benchmarks {'run successfully' if result.returncode == 0 else 'failed'}",
                    details={"output": result.stdout[:500], "stderr": result.stderr[:500]}
                ))
                
            except subprocess.TimeoutExpired:
                self.checks.append(ValidationCheck(
                    name="Benchmark Execution",
                    category="Performance",
                    passed=False,
                    message="❌ Benchmark execution timed out",
                    severity="WARNING"
                ))
        else:
            self.checks.append(ValidationCheck(
                name="Benchmark Script",
                category="Performance",
                passed=False,
                message="❌ Benchmark script missing",
                severity="WARNING"
            ))
    
    def _validate_ci_cd_workflows(self):
        """Validate CI/CD workflows are functional."""
        print("🔄 Validating CI/CD Workflows...")
        
        workflows_dir = self.project_root / ".github" / "workflows"
        if workflows_dir.exists():
            required_workflows = ["ci.yml", "release.yml", "security.yml", "docs.yml"]
            
            for workflow in required_workflows:
                workflow_path = workflows_dir / workflow
                if workflow_path.exists():
                    # Check workflow syntax
                    try:
                        with open(workflow_path) as f:
                            workflow_content = yaml.safe_load(f)
                            
                            # Basic workflow validation
                            has_name = "name" in workflow_content
                            has_on = "on" in workflow_content
                            has_jobs = "jobs" in workflow_content
                            
                            valid_workflow = has_name and has_on and has_jobs
                            
                            self.checks.append(ValidationCheck(
                                name=f"Workflow: {workflow}",
                                category="CI/CD",
                                passed=valid_workflow,
                                message=f"{'✅' if valid_workflow else '❌'} {workflow} {'is valid' if valid_workflow else 'has syntax issues'}"
                            ))
                    except Exception as e:
                        self.checks.append(ValidationCheck(
                            name=f"Workflow: {workflow}",
                            category="CI/CD",
                            passed=False,
                            message=f"❌ {workflow} has parsing errors: {str(e)}"
                        ))
                else:
                    self.checks.append(ValidationCheck(
                        name=f"Workflow: {workflow}",
                        category="CI/CD",
                        passed=False,
                        message=f"❌ {workflow} is missing"
                    ))
        else:
            self.checks.append(ValidationCheck(
                name="Workflows Directory",
                category="CI/CD",
                passed=False,
                message="❌ .github/workflows directory missing"
            ))
    
    def _validate_community_readiness(self):
        """Validate community engagement features."""
        print("👥 Validating Community Readiness...")
        
        community_files = [
            ("CONTRIBUTING.md", "Contribution guidelines"),
            ("CODE_OF_CONDUCT.md", "Code of conduct"),
            (".github/ISSUE_TEMPLATE/bug_report.yml", "Bug report template"),
            (".github/ISSUE_TEMPLATE/feature_request.yml", "Feature request template"),
            (".github/PULL_REQUEST_TEMPLATE.md", "Pull request template"),
            (".github/CODEOWNERS", "Code owners")
        ]
        
        for file_path, description in community_files:
            full_path = self.project_root / file_path
            self.checks.append(ValidationCheck(
                name=f"Community: {description}",
                category="Community",
                passed=full_path.exists(),
                message=f"{'✅' if full_path.exists() else '❌'} {description} {'exists' if full_path.exists() else 'missing'}"
            ))
    
    def _validate_marketing_assets(self):
        """Validate marketing and promotional materials."""
        print("🎯 Validating Marketing Assets...")
        
        marketing_files = [
            ("marketing/README.md", "Marketing guide"),
            ("marketing/badges_and_analytics.md", "Badges and analytics"),
            (".github/RELEASE_TEMPLATE.md", "Release template")
        ]
        
        for file_path, description in marketing_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                self.checks.append(ValidationCheck(
                    name=f"Marketing: {description}",
                    category="Marketing",
                    passed=size > 1000,  # At least 1KB of content
                    message=f"{'✅' if size > 1000 else '❌'} {description} {'exists with content' if size > 1000 else 'exists but too small'}",
                    severity="INFO"
                ))
            else:
                self.checks.append(ValidationCheck(
                    name=f"Marketing: {description}",
                    category="Marketing",
                    passed=False,
                    message=f"❌ {description} missing",
                    severity="WARNING"
                ))
    
    def _validate_legal_compliance(self):
        """Validate legal and licensing compliance."""
        print("⚖️ Validating Legal Compliance...")
        
        # Check LICENSE file
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
                
                self.checks.append(ValidationCheck(
                    name="License File",
                    category="Legal",
                    passed=detected_license is not None,
                    message=f"{'✅' if detected_license else '❌'} LICENSE file {'contains recognized license' if detected_license else 'unrecognized license format'}",
                    details={"license_type": detected_license}
                ))
        else:
            self.checks.append(ValidationCheck(
                name="License File",
                category="Legal",
                passed=False,
                message="❌ LICENSE file missing"
            ))
        
        # Check NOTICE file for attributions
        notice_path = self.project_root / "NOTICE"
        self.checks.append(ValidationCheck(
            name="Attribution File",
            category="Legal",
            passed=notice_path.exists(),
            message=f"{'✅' if notice_path.exists() else '❌'} NOTICE file {'exists' if notice_path.exists() else 'missing'}",
            severity="INFO"
        ))
        
        # Check for copyright headers in key files
        key_files = list(self.project_root.glob("src/**/*.py"))[:5]  # Check first 5 Python files
        files_with_copyright = 0
        
        for file_path in key_files:
            try:
                with open(file_path) as f:
                    content = f.read(500)  # Check first 500 chars
                    if "copyright" in content.lower() or "©" in content:
                        files_with_copyright += 1
            except Exception:
                continue
        
        self.checks.append(ValidationCheck(
            name="Copyright Headers",
            category="Legal",
            passed=files_with_copyright > 0,
            message=f"{'✅' if files_with_copyright > 0 else '❌'} Copyright headers found in {files_with_copyright}/{len(key_files)} checked files",
            severity="INFO"
        ))
    
    def _validate_release_artifacts(self):
        """Validate release artifacts can be built."""
        print("📦 Validating Release Artifacts...")
        
        # Check package build
        try:
            result = subprocess.run(
                ["python", "setup.py", "check"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            self.checks.append(ValidationCheck(
                name="Package Build Check",
                category="Release",
                passed=result.returncode == 0,
                message=f"{'✅' if result.returncode == 0 else '❌'} Package {'builds successfully' if result.returncode == 0 else 'has build issues'}",
                details={"stdout": result.stdout, "stderr": result.stderr}
            ))
            
        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Package Build Check",
                category="Release",
                passed=False,
                message=f"❌ Could not test package build: {str(e)}",
                severity="WARNING"
            ))
        
        # Check Docker build
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            try:
                result = subprocess.run(
                    ["docker", "build", "--dry-run", "."],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                self.checks.append(ValidationCheck(
                    name="Docker Build Check",
                    category="Release",
                    passed=result.returncode == 0,
                    message=f"{'✅' if result.returncode == 0 else '❌'} Docker {'builds successfully' if result.returncode == 0 else 'has build issues'}",
                    severity="INFO"
                ))
                
            except subprocess.TimeoutExpired:
                self.checks.append(ValidationCheck(
                    name="Docker Build Check",
                    category="Release",
                    passed=False,
                    message="❌ Docker build check timed out",
                    severity="WARNING"
                ))
            except FileNotFoundError:
                self.checks.append(ValidationCheck(
                    name="Docker Build Check",
                    category="Release",
                    passed=False,
                    message="❌ Docker not available for build check",
                    severity="INFO"
                ))
    
    def _generate_final_report(self) -> Tuple[bool, Dict[str, Any]]:
        """Generate comprehensive final validation report."""
        print("\n" + "=" * 60)
        print("🎯 FINAL VALIDATION REPORT")
        print("=" * 60)
        
        # Categorize results
        categories = {}
        for check in self.checks:
            if check.category not in categories:
                categories[check.category] = {"passed": 0, "failed": 0, "total": 0, "checks": []}
            
            categories[check.category]["total"] += 1
            categories[check.category]["checks"].append(check)
            
            if check.passed:
                categories[check.category]["passed"] += 1
            else:
                categories[check.category]["failed"] += 1
        
        # Calculate overall metrics
        total_checks = len(self.checks)
        passed_checks = sum(1 for c in self.checks if c.passed)
        failed_checks = total_checks - passed_checks
        success_rate = passed_checks / total_checks * 100 if total_checks > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"✅ Passed: {passed_checks}/{total_checks} checks ({success_rate:.1f}%)")
        print(f"❌ Failed: {failed_checks}")
        
        # Critical issues (blocking release)
        critical_issues = [c for c in self.checks if not c.passed and c.severity == "ERROR"]
        warnings = [c for c in self.checks if not c.passed and c.severity == "WARNING"]
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)} - MUST FIX):")
            for issue in critical_issues:
                print(f"❌ {issue.category}: {issue.name}")
                print(f"   {issue.message}")
                if issue.fix_suggestion:
                    print(f"   💡 Fix: {issue.fix_suggestion}")
                print()
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)} - SHOULD FIX):")
            for warning in warnings:
                print(f"⚠️  {warning.category}: {warning.name}")
                print(f"   {warning.message}")
                if warning.fix_suggestion:
                    print(f"   💡 Fix: {warning.fix_suggestion}")
                print()
        
        # Category breakdown
        print(f"\n📋 Category Breakdown:")
        for category, stats in categories.items():
            success_rate_cat = stats["passed"] / stats["total"] * 100
            status_icon = "✅" if stats["failed"] == 0 else "❌" if stats["failed"] > stats["passed"] else "⚠️"
            print(f"{status_icon} {category}: {stats['passed']}/{stats['total']} ({success_rate_cat:.0f}%)")
        
        # Generate detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(self.project_root.name),
            "validation_type": "final_release",
            "summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
                "success_rate": success_rate,
                "critical_issues": len(critical_issues),
                "warnings": len(warnings)
            },
            "categories": {
                category: {
                    "passed": stats["passed"],
                    "failed": stats["failed"], 
                    "total": stats["total"],
                    "success_rate": stats["passed"] / stats["total"] * 100
                }
                for category, stats in categories.items()
            },
            "checks": [
                {
                    "name": c.name,
                    "category": c.category,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                    "details": c.details,
                    "fix_suggestion": c.fix_suggestion
                }
                for c in self.checks
            ]
        }
        
        # Save report
        report_path = self.project_root / "final_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_path}")
        
        # Release readiness determination
        is_ready = len(critical_issues) == 0
        
        if is_ready:
            if len(warnings) == 0:
                print(f"\n🎉 REPOSITORY IS FULLY READY FOR RELEASE!")
                print("All checks passed - proceed with confidence! 🚀")
            else:
                print(f"\n✅ REPOSITORY IS READY FOR RELEASE!")
                print(f"⚠️  Consider addressing {len(warnings)} warnings for optimal quality.")
        else:
            print(f"\n❌ REPOSITORY IS NOT READY FOR RELEASE")
            print(f"🚨 {len(critical_issues)} critical issues must be resolved first.")
        
        return is_ready, report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final validation for BEM release")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    validator = FinalReleaseValidator(args.project_root)
    is_ready, report = validator.run_final_validation()
    
    if is_ready:
        print("\n🎯 FINAL VALIDATION: PASSED")
        sys.exit(0)
    else:
        print("\n❌ FINAL VALIDATION: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()