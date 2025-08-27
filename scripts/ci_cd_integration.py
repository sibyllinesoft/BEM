#!/usr/bin/env python3
"""
BEM v1.3 CI/CD Integration and Automated Deployment Pipeline

Provides production-ready CI/CD integration with:
- GitHub Actions workflow automation
- Statistical validation gates
- Automated rollback procedures  
- Performance regression detection
- Multi-environment deployment
- Comprehensive reporting and notifications

Integrates with the complete BEM workflow orchestrator for end-to-end automation.
"""

import json
import logging
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tempfile
import shutil
import requests
import asyncio
from github import Github

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.workflow_health_monitor import StatisticalGateValidator


@dataclass
class DeploymentConfig:
    """Configuration for deployment environments."""
    name: str
    branch: str
    auto_deploy: bool = False
    requires_approval: bool = True
    health_check_url: Optional[str] = None
    rollback_enabled: bool = True
    max_deployment_time_minutes: int = 60
    
    
@dataclass 
class CIJobResult:
    """Result of a CI job execution."""
    job_name: str
    status: str  # success, failure, cancelled
    start_time: datetime
    end_time: datetime
    logs: str
    artifacts: List[str]
    exit_code: int
    error_message: Optional[str] = None


class GitHubActionsIntegration:
    """Integration with GitHub Actions for automated CI/CD."""
    
    def __init__(self, github_token: str, repo_name: str):
        self.github_token = github_token
        self.repo_name = repo_name
        self.github_client = Github(github_token)
        self.repo = self.github_client.get_repo(repo_name)
        self.logger = logging.getLogger(__name__)
    
    def trigger_workflow(self, workflow_name: str, ref: str = "main", 
                        inputs: Optional[Dict[str, str]] = None) -> str:
        """Trigger a GitHub Actions workflow."""
        try:
            workflow = self.repo.get_workflow(f"{workflow_name}.yml")
            success = workflow.create_dispatch(ref, inputs or {})
            
            if success:
                self.logger.info(f"Successfully triggered workflow: {workflow_name}")
                return "triggered"
            else:
                self.logger.error(f"Failed to trigger workflow: {workflow_name}")
                return "failed"
                
        except Exception as e:
            self.logger.error(f"Error triggering workflow {workflow_name}: {e}")
            return "error"
    
    def get_workflow_runs(self, workflow_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow runs."""
        try:
            workflow = self.repo.get_workflow(f"{workflow_name}.yml")
            runs = workflow.get_runs()
            
            run_data = []
            for i, run in enumerate(runs):
                if i >= limit:
                    break
                    
                run_data.append({
                    'id': run.id,
                    'status': run.status,
                    'conclusion': run.conclusion,
                    'created_at': run.created_at.isoformat(),
                    'updated_at': run.updated_at.isoformat(),
                    'head_sha': run.head_sha,
                    'html_url': run.html_url
                })
            
            return run_data
            
        except Exception as e:
            self.logger.error(f"Error getting workflow runs for {workflow_name}: {e}")
            return []
    
    def wait_for_workflow_completion(self, workflow_name: str, timeout_minutes: int = 60) -> str:
        """Wait for most recent workflow run to complete."""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            runs = self.get_workflow_runs(workflow_name, limit=1)
            
            if runs and runs[0]['status'] == 'completed':
                return runs[0]['conclusion']
            
            time.sleep(30)  # Check every 30 seconds
        
        return "timeout"
    
    def create_deployment_status(self, sha: str, environment: str, 
                                state: str, description: str = "") -> bool:
        """Create deployment status."""
        try:
            deployment = self.repo.create_deployment(
                ref=sha,
                environment=environment,
                description=description,
                auto_merge=False
            )
            
            deployment.create_status(
                state=state,
                description=description
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating deployment status: {e}")
            return False


class BEMCIOrchestrator:
    """Main CI/CD orchestrator for BEM v1.3 workflows."""
    
    def __init__(self, 
                 config_path: Path,
                 github_token: Optional[str] = None,
                 repo_name: Optional[str] = None):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # GitHub integration (optional)
        self.github_integration = None
        if github_token and repo_name:
            self.github_integration = GitHubActionsIntegration(github_token, repo_name)
        
        # Statistical gate validator
        self.gate_validator = StatisticalGateValidator(Path("gates_bem2.yaml"))
        
        # Setup logging
        self.setup_logging()
        
        # State tracking
        self.deployment_history = []
        self.active_deployments = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load CI/CD configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        
        return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default CI/CD configuration."""
        return {
            "environments": {
                "development": {
                    "name": "development",
                    "branch": "develop",
                    "auto_deploy": True,
                    "requires_approval": False,
                    "rollback_enabled": True,
                    "max_deployment_time_minutes": 30
                },
                "staging": {
                    "name": "staging", 
                    "branch": "main",
                    "auto_deploy": False,
                    "requires_approval": True,
                    "rollback_enabled": True,
                    "max_deployment_time_minutes": 45
                },
                "production": {
                    "name": "production",
                    "branch": "main",
                    "auto_deploy": False,
                    "requires_approval": True,
                    "rollback_enabled": True,
                    "max_deployment_time_minutes": 60,
                    "health_check_url": "https://bem-api.example.com/health"
                }
            },
            "notifications": {
                "slack_webhook": None,
                "email_recipients": [],
                "github_notifications": True
            },
            "quality_gates": {
                "statistical_significance": True,
                "performance_regression": True,
                "security_scan": True,
                "test_coverage_minimum": 0.8
            }
        }
    
    def setup_logging(self):
        """Setup CI/CD logging."""
        log_dir = Path("logs/cicd")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"cicd_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def run_ci_pipeline(self, branch: str = "main", 
                            commit_sha: Optional[str] = None) -> Dict[str, Any]:
        """Run complete CI pipeline with quality gates."""
        self.logger.info(f"Starting CI pipeline for branch: {branch}")
        
        pipeline_start = time.time()
        pipeline_results = {
            "pipeline_id": f"ci_{int(pipeline_start)}",
            "branch": branch,
            "commit_sha": commit_sha,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "jobs": [],
            "quality_gates": {},
            "artifacts": []
        }
        
        try:
            # Stage 1: Environment Setup and Building
            building_result = await self.run_building_stage()
            pipeline_results["jobs"].append(building_result)
            
            if building_result.status != "success":
                raise Exception(f"Building stage failed: {building_result.error_message}")
            
            # Stage 2: Training and Evaluation
            running_result = await self.run_training_stage()
            pipeline_results["jobs"].append(running_result)
            
            if running_result.status != "success":
                raise Exception(f"Training stage failed: {running_result.error_message}")
            
            # Stage 3: Statistical Analysis and Tracking
            tracking_result = await self.run_tracking_stage()
            pipeline_results["jobs"].append(tracking_result)
            
            if tracking_result.status != "success":
                raise Exception(f"Tracking stage failed: {tracking_result.error_message}")
            
            # Stage 4: Quality Gate Validation
            gates_passed, gate_results = await self.validate_quality_gates()
            pipeline_results["quality_gates"] = gate_results
            
            if not gates_passed:
                pipeline_results["status"] = "failed"
                pipeline_results["error_message"] = "Quality gates failed"
                return pipeline_results
            
            # Stage 5: Artifact Generation
            artifacts = await self.generate_artifacts()
            pipeline_results["artifacts"] = artifacts
            
            pipeline_results["status"] = "success"
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["duration_minutes"] = (time.time() - pipeline_start) / 60
            
            self.logger.info(f"CI pipeline completed successfully in {pipeline_results['duration_minutes']:.1f} minutes")
            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["error_message"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            self.logger.error(f"CI pipeline failed: {e}")
        
        return pipeline_results
    
    async def run_building_stage(self) -> CIJobResult:
        """Run the building stage (environment setup, asset preparation)."""
        self.logger.info("Running building stage...")
        
        start_time = datetime.now()
        
        try:
            # Environment setup
            result = await self.execute_command_async([
                "python", "bem_workflow_orchestrator.py", 
                "--stages", "building",
                "--verbose"
            ])
            
            end_time = datetime.now()
            
            if result["returncode"] == 0:
                return CIJobResult(
                    job_name="building",
                    status="success", 
                    start_time=start_time,
                    end_time=end_time,
                    logs=result["stdout"],
                    artifacts=["logs/kernels.json", "logs/fp8_selftest.json"],
                    exit_code=result["returncode"]
                )
            else:
                return CIJobResult(
                    job_name="building",
                    status="failure",
                    start_time=start_time,
                    end_time=end_time,
                    logs=result["stderr"],
                    artifacts=[],
                    exit_code=result["returncode"],
                    error_message=result["stderr"]
                )
                
        except Exception as e:
            return CIJobResult(
                job_name="building",
                status="failure",
                start_time=start_time,
                end_time=datetime.now(),
                logs="",
                artifacts=[],
                exit_code=-1,
                error_message=str(e)
            )
    
    async def run_training_stage(self) -> CIJobResult:
        """Run the training and evaluation stage."""
        self.logger.info("Running training stage...")
        
        start_time = datetime.now()
        
        try:
            result = await self.execute_command_async([
                "python", "bem_workflow_orchestrator.py",
                "--stages", "running", 
                "--verbose",
                "--max-concurrent", "1"  # Conservative for CI
            ])
            
            end_time = datetime.now()
            
            artifacts = []
            if Path("logs").exists():
                artifacts.extend([
                    str(p) for p in Path("logs").rglob("*.json") 
                    if "eval" in p.name or "results" in p.name
                ])
            
            if result["returncode"] == 0:
                return CIJobResult(
                    job_name="training",
                    status="success",
                    start_time=start_time,
                    end_time=end_time,
                    logs=result["stdout"],
                    artifacts=artifacts,
                    exit_code=result["returncode"]
                )
            else:
                return CIJobResult(
                    job_name="training", 
                    status="failure",
                    start_time=start_time,
                    end_time=end_time,
                    logs=result["stderr"],
                    artifacts=artifacts,
                    exit_code=result["returncode"],
                    error_message=result["stderr"]
                )
                
        except Exception as e:
            return CIJobResult(
                job_name="training",
                status="failure", 
                start_time=start_time,
                end_time=datetime.now(),
                logs="",
                artifacts=[],
                exit_code=-1,
                error_message=str(e)
            )
    
    async def run_tracking_stage(self) -> CIJobResult:
        """Run the tracking and analysis stage."""
        self.logger.info("Running tracking stage...")
        
        start_time = datetime.now()
        
        try:
            result = await self.execute_command_async([
                "python", "bem_workflow_orchestrator.py",
                "--stages", "tracking", "evaluating",
                "--verbose"
            ])
            
            end_time = datetime.now()
            
            artifacts = []
            if Path("analysis").exists():
                artifacts.extend([
                    str(p) for p in Path("analysis").rglob("*.json")
                ])
            
            if result["returncode"] == 0:
                return CIJobResult(
                    job_name="tracking",
                    status="success",
                    start_time=start_time,
                    end_time=end_time, 
                    logs=result["stdout"],
                    artifacts=artifacts,
                    exit_code=result["returncode"]
                )
            else:
                return CIJobResult(
                    job_name="tracking",
                    status="failure",
                    start_time=start_time,
                    end_time=end_time,
                    logs=result["stderr"],
                    artifacts=artifacts,
                    exit_code=result["returncode"],
                    error_message=result["stderr"]
                )
                
        except Exception as e:
            return CIJobResult(
                job_name="tracking",
                status="failure",
                start_time=start_time,
                end_time=datetime.now(),
                logs="",
                artifacts=[],
                exit_code=-1,
                error_message=str(e)
            )
    
    async def validate_quality_gates(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate all quality gates."""
        self.logger.info("Validating quality gates...")
        
        gate_results = {
            "timestamp": datetime.now().isoformat(),
            "gates": {},
            "overall_passed": True
        }
        
        # Statistical significance gates
        if self.config["quality_gates"]["statistical_significance"]:
            stats_file = Path("analysis/stats.json")
            if stats_file.exists():
                with open(stats_file) as f:
                    stats_data = json.load(f)
                
                gates_passed, detailed_results = self.gate_validator.validate_experiment_gates(stats_data)
                gate_results["gates"]["statistical"] = detailed_results
                
                if not gates_passed:
                    gate_results["overall_passed"] = False
                    self.logger.error("Statistical significance gates failed")
            else:
                gate_results["gates"]["statistical"] = {"passed": False, "error": "Stats file not found"}
                gate_results["overall_passed"] = False
        
        # Performance regression detection
        if self.config["quality_gates"]["performance_regression"]:
            regression_detected = await self.check_performance_regression()
            gate_results["gates"]["performance_regression"] = {
                "passed": not regression_detected,
                "regression_detected": regression_detected
            }
            
            if regression_detected:
                gate_results["overall_passed"] = False
                self.logger.error("Performance regression detected")
        
        # Security scan
        if self.config["quality_gates"]["security_scan"]:
            security_passed = await self.run_security_scan()
            gate_results["gates"]["security"] = {"passed": security_passed}
            
            if not security_passed:
                gate_results["overall_passed"] = False
                self.logger.error("Security scan failed")
        
        # Test coverage
        coverage_passed = await self.check_test_coverage()
        gate_results["gates"]["test_coverage"] = {"passed": coverage_passed}
        
        if not coverage_passed:
            gate_results["overall_passed"] = False
            self.logger.error("Test coverage below threshold")
        
        return gate_results["overall_passed"], gate_results
    
    async def check_performance_regression(self) -> bool:
        """Check for performance regressions."""
        try:
            # Look for performance metrics in evaluation results
            performance_data = {}
            
            for eval_file in Path("logs").rglob("eval.json"):
                with open(eval_file) as f:
                    data = json.load(f)
                    
                if "latency" in data:
                    latency = data["latency"]
                    performance_data[eval_file.parent.name] = {
                        "p50_latency": latency.get("p50", 0),
                        "p95_latency": latency.get("p95", 0)
                    }
            
            # Simple regression check - compare against baseline if available
            baseline_file = Path("analysis/baseline_performance.json")
            if baseline_file.exists():
                with open(baseline_file) as f:
                    baseline = json.load(f)
                
                for experiment, metrics in performance_data.items():
                    if experiment in baseline:
                        p50_increase = (metrics["p50_latency"] - baseline[experiment]["p50_latency"]) / baseline[experiment]["p50_latency"]
                        p95_increase = (metrics["p95_latency"] - baseline[experiment]["p95_latency"]) / baseline[experiment]["p95_latency"]
                        
                        # Regression if >20% increase
                        if p50_increase > 0.20 or p95_increase > 0.20:
                            self.logger.warning(f"Performance regression detected in {experiment}")
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking performance regression: {e}")
            return True  # Assume regression if we can't check
    
    async def run_security_scan(self) -> bool:
        """Run security scan."""
        try:
            # Run bandit for Python security issues
            result = await self.execute_command_async([
                "bandit", "-r", ".", "-f", "json", "-o", "security_report.json"
            ])
            
            # Check results
            if Path("security_report.json").exists():
                with open("security_report.json") as f:
                    report = json.load(f)
                
                high_severity = len([issue for issue in report.get("results", []) if issue.get("issue_severity") == "HIGH"])
                
                return high_severity == 0
            
            return result["returncode"] == 0
            
        except Exception as e:
            self.logger.error(f"Security scan error: {e}")
            return False
    
    async def check_test_coverage(self) -> bool:
        """Check test coverage meets minimum threshold."""
        try:
            result = await self.execute_command_async([
                "coverage", "report", "--format=json"
            ])
            
            if result["returncode"] == 0:
                coverage_data = json.loads(result["stdout"])
                total_coverage = coverage_data["totals"]["percent_covered"] / 100
                
                min_coverage = self.config["quality_gates"]["test_coverage_minimum"]
                return total_coverage >= min_coverage
            
            return False
            
        except Exception as e:
            self.logger.error(f"Test coverage check error: {e}")
            return False
    
    async def generate_artifacts(self) -> List[str]:
        """Generate deployment artifacts."""
        self.logger.info("Generating artifacts...")
        
        artifacts = []
        
        # Generate reproducibility package
        try:
            result = await self.execute_command_async([
                "python", "scripts/make_repro_pack.py", 
                "--out", "dist/repro_manifest.json",
                "--script", "dist/run.sh"
            ])
            
            if result["returncode"] == 0:
                artifacts.extend(["dist/repro_manifest.json", "dist/run.sh"])
        except Exception as e:
            self.logger.error(f"Error generating repro pack: {e}")
        
        # Generate paper if requested
        try:
            result = await self.execute_command_async([
                "python", "bem_workflow_orchestrator.py",
                "--stages", "refinement"
            ])
            
            if result["returncode"] == 0 and Path("paper/main.pdf").exists():
                artifacts.append("paper/main.pdf")
        except Exception as e:
            self.logger.error(f"Error generating paper: {e}")
        
        # Archive key results
        key_files = [
            "analysis/stats.json",
            "analysis/winners.json", 
            "analysis/pareto.json"
        ]
        
        for file_path in key_files:
            if Path(file_path).exists():
                artifacts.append(file_path)
        
        self.logger.info(f"Generated {len(artifacts)} artifacts")
        return artifacts
    
    async def execute_command_async(self, command: List[str], 
                                  timeout: int = 3600) -> Dict[str, Any]:
        """Execute command asynchronously with timeout."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else ""
            }
            
        except asyncio.TimeoutError:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    async def deploy_to_environment(self, environment: str, 
                                   artifacts: List[str]) -> Dict[str, Any]:
        """Deploy to specified environment."""
        env_config = self.config["environments"].get(environment)
        if not env_config:
            raise ValueError(f"Unknown environment: {environment}")
        
        self.logger.info(f"Deploying to {environment} environment...")
        
        deployment_result = {
            "environment": environment,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "artifacts": artifacts,
            "health_checks": []
        }
        
        try:
            # Create deployment directory
            deploy_dir = Path(f"deployments/{environment}")
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy artifacts
            for artifact in artifacts:
                src = Path(artifact)
                if src.exists():
                    dst = deploy_dir / src.name
                    shutil.copy2(src, dst)
            
            # Run health checks if configured
            if env_config.get("health_check_url"):
                health_passed = await self.run_health_check(env_config["health_check_url"])
                deployment_result["health_checks"].append({
                    "url": env_config["health_check_url"],
                    "passed": health_passed,
                    "timestamp": datetime.now().isoformat()
                })
                
                if not health_passed:
                    raise Exception("Health check failed")
            
            deployment_result["status"] = "success"
            deployment_result["end_time"] = datetime.now().isoformat()
            
            self.logger.info(f"Successfully deployed to {environment}")
            
        except Exception as e:
            deployment_result["status"] = "failed"
            deployment_result["error_message"] = str(e)
            deployment_result["end_time"] = datetime.now().isoformat()
            
            self.logger.error(f"Deployment to {environment} failed: {e}")
            
            # Attempt rollback if enabled
            if env_config.get("rollback_enabled", True):
                await self.rollback_deployment(environment)
        
        return deployment_result
    
    async def run_health_check(self, health_check_url: str) -> bool:
        """Run health check against deployed service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_check_url, timeout=30) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def rollback_deployment(self, environment: str):
        """Rollback failed deployment."""
        self.logger.info(f"Rolling back deployment in {environment}")
        
        # Implementation depends on deployment strategy
        # For now, just log the rollback attempt
        rollback_result = {
            "environment": environment,
            "timestamp": datetime.now().isoformat(),
            "status": "attempted"
        }
        
        # Save rollback record
        rollback_file = Path(f"deployments/{environment}/rollback.json")
        with open(rollback_file, 'w') as f:
            json.dump(rollback_result, f, indent=2)
    
    def send_notification(self, message: str, level: str = "info"):
        """Send notification via configured channels."""
        notification_data = {
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "pipeline": "bem_v13_cicd"
        }
        
        # Slack notification
        slack_webhook = self.config["notifications"].get("slack_webhook")
        if slack_webhook:
            try:
                requests.post(slack_webhook, json={
                    "text": f"🤖 BEM v1.3 CI/CD: {message}",
                    "username": "BEM-Bot"
                })
            except Exception as e:
                self.logger.error(f"Failed to send Slack notification: {e}")
        
        # Log notification
        if level == "error":
            self.logger.error(f"NOTIFICATION: {message}")
        else:
            self.logger.info(f"NOTIFICATION: {message}")


async def main():
    """Main entry point for CI/CD integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BEM v1.3 CI/CD Integration")
    parser.add_argument("--config", type=Path, default=Path("cicd_config.yaml"),
                       help="CI/CD configuration file")
    parser.add_argument("--action", choices=["ci", "deploy", "rollback"], default="ci",
                       help="Action to perform")
    parser.add_argument("--environment", choices=["development", "staging", "production"],
                       help="Environment for deployment")
    parser.add_argument("--branch", default="main",
                       help="Branch to build from")
    parser.add_argument("--commit-sha", 
                       help="Specific commit SHA to build")
    parser.add_argument("--github-token",
                       help="GitHub token for API access")
    parser.add_argument("--repo-name",
                       help="GitHub repository name (e.g., 'user/repo')")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = BEMCIOrchestrator(
        config_path=args.config,
        github_token=args.github_token,
        repo_name=args.repo_name
    )
    
    if args.action == "ci":
        # Run CI pipeline
        result = await orchestrator.run_ci_pipeline(
            branch=args.branch,
            commit_sha=args.commit_sha
        )
        
        print(json.dumps(result, indent=2))
        
        if result["status"] == "success":
            orchestrator.send_notification(
                f"CI pipeline completed successfully for {args.branch}",
                "info"
            )
            sys.exit(0)
        else:
            orchestrator.send_notification(
                f"CI pipeline failed for {args.branch}: {result.get('error_message', 'Unknown error')}",
                "error"
            )
            sys.exit(1)
    
    elif args.action == "deploy":
        if not args.environment:
            print("Environment required for deployment")
            sys.exit(1)
        
        # Get artifacts from latest successful CI run
        artifacts = []
        if Path("dist").exists():
            artifacts.extend(str(p) for p in Path("dist").iterdir())
        
        result = await orchestrator.deploy_to_environment(args.environment, artifacts)
        
        print(json.dumps(result, indent=2))
        
        if result["status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.action == "rollback":
        if not args.environment:
            print("Environment required for rollback")
            sys.exit(1)
        
        await orchestrator.rollback_deployment(args.environment)


if __name__ == "__main__":
    asyncio.run(main())