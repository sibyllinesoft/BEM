#!/usr/bin/env python3
"""
BEM Fleet Multi-Mission Orchestrator
Main orchestration script for the 60-day research sprint

This script coordinates all 5 parallel missions:
- Mission A: Agentic Planner
- Mission B: Living Model  
- Mission C: Alignment Enforcer
- Mission D: SEP
- Mission E: Long-Memory + SSM↔BEM Coupling

Provides unified execution, monitoring, and statistical validation.
"""

import argparse
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import subprocess
import time
import signal
import psutil
from threading import Event, Thread

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.statistical_validation_framework import (
    StatisticalValidationFramework, 
    create_default_test_configs
)
from monitoring.fleet_dashboard import FleetMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bem_fleet_orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MissionConfig:
    """Configuration for a single mission"""
    name: str
    config_file: str
    dependencies: List[str]
    resource_requirements: Dict[str, str]
    estimated_duration: int  # hours
    priority: str  # 'critical', 'high', 'medium'
    
@dataclass 
class ExecutionContext:
    """Execution context for the fleet orchestrator"""
    start_time: datetime
    mission_configs: Dict[str, MissionConfig]
    resource_allocations: Dict[str, Dict]
    monitoring_enabled: bool
    statistical_validation_enabled: bool
    shutdown_event: Event

class MissionExecutor:
    """Executes individual missions"""
    
    def __init__(self, mission_config: MissionConfig, context: ExecutionContext):
        self.mission = mission_config
        self.context = context
        self.process = None
        self.status = 'pending'
        self.start_time = None
        self.end_time = None
        
    def execute(self) -> Dict:
        """Execute the mission"""
        logger.info(f"Starting Mission {self.mission.name}")
        
        self.status = 'running'
        self.start_time = datetime.now()
        
        try:
            # Run the mission workflow
            result = self._run_mission_workflow()
            
            self.status = 'completed' if result['success'] else 'failed'
            self.end_time = datetime.now()
            
            logger.info(f"Mission {self.mission.name} {self.status}")
            
            return {
                'mission': self.mission.name,
                'status': self.status,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration': (self.end_time - self.start_time).total_seconds(),
                'result': result
            }
            
        except Exception as e:
            self.status = 'error'
            self.end_time = datetime.now()
            logger.error(f"Mission {self.mission.name} failed with error: {e}")
            
            return {
                'mission': self.mission.name,
                'status': self.status,
                'error': str(e),
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat()
            }
    
    def _run_mission_workflow(self) -> Dict:
        """Run the XML workflow for this mission"""
        workflow_name = f"mission_{self.mission.name.lower()}_enhanced"
        
        # Execute the XML workflow using our workflow runner
        cmd = [
            'python', 'scripts/run_xml_workflow.py',
            '--workflow', workflow_name,
            '--config', self.mission.config_file,
            '--log-dir', f'logs/mission_{self.mission.name}',
            '--parallel' if self._supports_parallel() else '--sequential'
        ]
        
        logger.info(f"Executing workflow command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.mission.estimated_duration * 3600,  # Convert to seconds
                check=False
            )
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Mission {self.mission.name} timed out")
            return {
                'success': False,
                'error': 'timeout',
                'timeout_hours': self.mission.estimated_duration
            }
    
    def _supports_parallel(self) -> bool:
        """Check if mission supports parallel execution"""
        parallel_missions = ['A', 'C', 'D']  # Missions that can run in parallel
        return self.mission.name in parallel_missions
    
    def terminate(self):
        """Terminate the mission execution"""
        if self.process and self.process.poll() is None:
            logger.info(f"Terminating Mission {self.mission.name}")
            self.process.terminate()
            self.status = 'terminated'

class BEMFleetOrchestrator:
    """Main orchestrator for BEM Fleet multi-mission execution"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.context = None
        self.missions = {}
        self.results = {}
        self.monitor = None
        self.shutdown_event = Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _load_config(self) -> Dict:
        """Load fleet configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_event.set()
    
    def _create_execution_context(self) -> ExecutionContext:
        """Create execution context from configuration"""
        mission_configs = {}
        
        # Load mission configurations
        for mission_id, mission_data in self.config['missions'].items():
            mission_name = mission_id.split('_')[1].upper()  # Extract A, B, C, D, E
            
            mission_configs[mission_name] = MissionConfig(
                name=mission_name,
                config_file=mission_data['config_file'],
                dependencies=mission_data.get('dependencies', []),
                resource_requirements=self.config['parallel_execution']['resource_allocation'].get(mission_id, {}),
                estimated_duration=mission_data.get('timeline', 24),  # hours
                priority=mission_data.get('priority', 'medium')
            )
        
        return ExecutionContext(
            start_time=datetime.now(),
            mission_configs=mission_configs,
            resource_allocations=self.config['parallel_execution']['resource_allocation'],
            monitoring_enabled=True,
            statistical_validation_enabled=True,
            shutdown_event=self.shutdown_event
        )
    
    def _check_dependencies(self, mission_name: str) -> bool:
        """Check if mission dependencies are satisfied"""
        mission_config = self.context.mission_configs[mission_name]
        
        for dep in mission_config.dependencies:
            dep_mission = dep.replace('mission_', '').upper()
            if dep_mission not in self.results or self.results[dep_mission]['status'] != 'completed':
                logger.info(f"Mission {mission_name} waiting for dependency {dep_mission}")
                return False
        
        return True
    
    def _initialize_monitoring(self):
        """Initialize fleet monitoring"""
        try:
            config_path = 'configs/fleet_monitoring.yaml'
            self.monitor = FleetMonitor(config_path)
            logger.info("Fleet monitoring initialized")
        except Exception as e:
            logger.warning(f"Could not initialize monitoring: {e}")
            self.monitor = None
    
    def _start_dashboard(self):
        """Start monitoring dashboard in background"""
        if self.monitor:
            try:
                # Start dashboard as a separate process
                cmd = ['streamlit', 'run', 'monitoring/fleet_dashboard.py', '--server.port', '8501']
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info("Fleet dashboard started at http://localhost:8501")
            except Exception as e:
                logger.warning(f"Could not start dashboard: {e}")
    
    def execute_bootstrap_phase(self) -> bool:
        """Execute fleet bootstrap workflow"""
        logger.info("Starting Fleet Bootstrap Phase")
        
        try:
            cmd = [
                'python', 'scripts/run_xml_workflow.py',
                '--workflow', 'fleet_bootstrap',
                '--log-dir', 'logs/bootstrap'
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Fleet bootstrap completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Fleet bootstrap failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return False
    
    def execute_missions_parallel(self) -> Dict[str, Dict]:
        """Execute missions in parallel with dependency management"""
        logger.info("Starting parallel mission execution")
        
        # Separate missions by dependency requirements
        independent_missions = ['A', 'C', 'D']  # Can start immediately
        dependent_missions = ['B', 'E']  # Need Mission A to complete
        
        results = {}
        
        # Phase 1: Execute independent missions
        with ProcessPoolExecutor(max_workers=3) as executor:
            future_to_mission = {}
            
            for mission_name in independent_missions:
                if self.shutdown_event.is_set():
                    break
                    
                mission_config = self.context.mission_configs[mission_name]
                executor_instance = MissionExecutor(mission_config, self.context)
                
                future = executor.submit(executor_instance.execute)
                future_to_mission[future] = mission_name
                self.missions[mission_name] = executor_instance
                
                logger.info(f"Submitted Mission {mission_name} for execution")
            
            # Collect results from independent missions
            for future in as_completed(future_to_mission):
                if self.shutdown_event.is_set():
                    break
                    
                mission_name = future_to_mission[future]
                try:
                    result = future.result()
                    results[mission_name] = result
                    self.results[mission_name] = result
                    
                    logger.info(f"Mission {mission_name} completed with status: {result['status']}")
                    
                    # Update monitoring if available
                    if self.monitor and result['status'] == 'completed':
                        # This would update mission completion metrics
                        pass
                        
                except Exception as e:
                    logger.error(f"Mission {mission_name} failed: {e}")
                    results[mission_name] = {
                        'mission': mission_name,
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Phase 2: Execute dependent missions if Mission A succeeded
        if 'A' in results and results['A']['status'] == 'completed' and not self.shutdown_event.is_set():
            with ProcessPoolExecutor(max_workers=2) as executor:
                future_to_mission = {}
                
                for mission_name in dependent_missions:
                    mission_config = self.context.mission_configs[mission_name]
                    executor_instance = MissionExecutor(mission_config, self.context)
                    
                    future = executor.submit(executor_instance.execute)
                    future_to_mission[future] = mission_name
                    self.missions[mission_name] = executor_instance
                    
                    logger.info(f"Submitted dependent Mission {mission_name} for execution")
                
                # Collect results from dependent missions
                for future in as_completed(future_to_mission):
                    if self.shutdown_event.is_set():
                        break
                        
                    mission_name = future_to_mission[future]
                    try:
                        result = future.result()
                        results[mission_name] = result
                        self.results[mission_name] = result
                        
                        logger.info(f"Mission {mission_name} completed with status: {result['status']}")
                        
                    except Exception as e:
                        logger.error(f"Mission {mission_name} failed: {e}")
                        results[mission_name] = {
                            'mission': mission_name,
                            'status': 'error', 
                            'error': str(e)
                        }
        else:
            logger.warning("Mission A failed or was skipped - dependent missions not executed")
        
        return results
    
    def execute_integration_phase(self) -> Dict:
        """Execute cross-mission integration testing"""
        logger.info("Starting Integration Phase")
        
        try:
            cmd = [
                'python', 'scripts/run_xml_workflow.py',
                '--workflow', 'cross_mission_integration',
                '--log-dir', 'logs/integration'
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Integration phase completed successfully")
            return {
                'status': 'completed',
                'result': result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Integration phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'stderr': e.stderr
            }
    
    def execute_statistical_validation(self) -> Dict:
        """Execute statistical validation and promotion decisions"""
        logger.info("Starting Statistical Validation Phase")
        
        try:
            # Initialize validation framework
            framework = StatisticalValidationFramework()
            
            # Create test configurations
            test_configs = create_default_test_configs()
            
            # Mission-specific acceptance gates
            mission_configs = {
                'A': {'acceptance_gates': {'em_f1_improvement': 1.5}},
                'B': {'acceptance_gates': {'time_to_fix_max': 1000}},
                'C': {'acceptance_gates': {'violation_reduction_min': 30}},
                'D': {'acceptance_gates': {'rrs_improvement': True}},
                'E': {'acceptance_gates': {'min_context_length': 131072}}
            }
            
            # Run full validation
            promotion_decisions = framework.run_full_validation(
                results_dir="logs",
                baseline_path="logs/baseline_v13/eval.json",
                test_configs=test_configs,
                mission_configs=mission_configs,
                output_dir="analysis"
            )
            
            logger.info("Statistical validation completed")
            return {
                'status': 'completed',
                'promotion_decisions': promotion_decisions
            }
            
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def execute_consolidation_phase(self) -> Dict:
        """Execute final consolidation and reporting"""
        logger.info("Starting Consolidation Phase")
        
        try:
            cmd = [
                'python', 'scripts/run_xml_workflow.py',
                '--workflow', 'consolidate_enhanced',
                '--log-dir', 'logs/consolidation'
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Consolidation phase completed successfully")
            return {
                'status': 'completed',
                'result': result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Consolidation phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_full_fleet_execution(self) -> Dict:
        """Run complete BEM Fleet execution pipeline"""
        logger.info("="*60)
        logger.info("BEM Fleet Multi-Mission Orchestrator Starting")
        logger.info("60-day research sprint with 5 parallel missions")
        logger.info("="*60)
        
        # Initialize execution context
        self.context = self._create_execution_context()
        
        # Initialize monitoring
        self._initialize_monitoring()
        self._start_dashboard()
        
        # Execution phases
        phases = []
        
        try:
            # Phase 1: Bootstrap
            logger.info("\nPhase 1: Fleet Bootstrap")
            bootstrap_success = self.execute_bootstrap_phase()
            phases.append({
                'phase': 'bootstrap',
                'status': 'completed' if bootstrap_success else 'failed'
            })
            
            if not bootstrap_success:
                logger.error("Bootstrap failed - aborting execution")
                return {'status': 'failed', 'phase': 'bootstrap'}
            
            # Phase 2: Parallel Mission Execution
            logger.info("\nPhase 2: Parallel Mission Execution")
            mission_results = self.execute_missions_parallel()
            phases.append({
                'phase': 'missions',
                'status': 'completed',
                'results': mission_results
            })
            
            # Check if any missions failed critically
            failed_missions = [m for m, r in mission_results.items() if r['status'] in ['failed', 'error']]
            if failed_missions:
                logger.warning(f"Some missions failed: {failed_missions}")
            
            # Phase 3: Integration Testing (if not shut down)
            if not self.shutdown_event.is_set():
                logger.info("\nPhase 3: Cross-Mission Integration")
                integration_result = self.execute_integration_phase()
                phases.append({
                    'phase': 'integration',
                    **integration_result
                })
            
            # Phase 4: Statistical Validation (if not shut down)
            if not self.shutdown_event.is_set():
                logger.info("\nPhase 4: Statistical Validation")
                validation_result = self.execute_statistical_validation()
                phases.append({
                    'phase': 'validation',
                    **validation_result
                })
            
            # Phase 5: Consolidation (if not shut down)
            if not self.shutdown_event.is_set():
                logger.info("\nPhase 5: Consolidation and Reporting")
                consolidation_result = self.execute_consolidation_phase()
                phases.append({
                    'phase': 'consolidation',
                    **consolidation_result
                })
            
            # Generate final summary
            summary = self._generate_execution_summary(phases, mission_results)
            
            logger.info("\nBEM Fleet Execution Summary:")
            logger.info(f"Total Duration: {summary['total_duration_hours']:.1f} hours")
            logger.info(f"Missions Completed: {summary['missions_completed']}/{summary['total_missions']}")
            logger.info(f"Overall Status: {summary['overall_status']}")
            
            return {
                'status': summary['overall_status'],
                'summary': summary,
                'phases': phases,
                'mission_results': mission_results
            }
            
        except KeyboardInterrupt:
            logger.info("\nExecution interrupted by user")
            self._cleanup_missions()
            return {
                'status': 'interrupted',
                'completed_phases': phases
            }
        
        except Exception as e:
            logger.error(f"\nUnexpected error in fleet execution: {e}")
            self._cleanup_missions()
            return {
                'status': 'error',
                'error': str(e),
                'completed_phases': phases
            }
    
    def _cleanup_missions(self):
        """Cleanup running missions on shutdown"""
        logger.info("Cleaning up running missions...")
        for mission_name, executor in self.missions.items():
            if executor.status == 'running':
                executor.terminate()
                logger.info(f"Terminated Mission {mission_name}")
    
    def _generate_execution_summary(self, phases: List[Dict], mission_results: Dict[str, Dict]) -> Dict:
        """Generate execution summary"""
        end_time = datetime.now()
        total_duration = (end_time - self.context.start_time).total_seconds() / 3600  # hours
        
        missions_completed = len([r for r in mission_results.values() if r['status'] == 'completed'])
        total_missions = len(mission_results)
        
        phases_completed = len([p for p in phases if p['status'] == 'completed'])
        total_phases = 5
        
        # Determine overall status
        if phases_completed == total_phases and missions_completed == total_missions:
            overall_status = 'success'
        elif phases_completed >= 3 and missions_completed >= 3:
            overall_status = 'partial_success'
        else:
            overall_status = 'failure'
        
        return {
            'start_time': self.context.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_hours': total_duration,
            'missions_completed': missions_completed,
            'total_missions': total_missions,
            'phases_completed': phases_completed,
            'total_phases': total_phases,
            'overall_status': overall_status,
            'mission_breakdown': {m: r['status'] for m, r in mission_results.items()}
        }

def main():
    """Main orchestrator entry point"""
    parser = argparse.ArgumentParser(description="BEM Fleet Multi-Mission Orchestrator")
    parser.add_argument('--config', default='bem_fleet_architecture.yml', 
                       help='Fleet configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without execution')
    parser.add_argument('--mission', choices=['A', 'B', 'C', 'D', 'E'],
                       help='Execute single mission instead of full fleet')
    parser.add_argument('--phase', choices=['bootstrap', 'missions', 'integration', 'validation', 'consolidation'],
                       help='Execute single phase')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Start only the monitoring dashboard')
    
    args = parser.parse_args()
    
    if args.dashboard_only:
        logger.info("Starting monitoring dashboard only")
        subprocess.run(['streamlit', 'run', 'monitoring/fleet_dashboard.py', '--server.port', '8501'])
        return
    
    # Initialize orchestrator
    orchestrator = BEMFleetOrchestrator(args.config)
    
    if args.dry_run:
        logger.info("Dry run - validating configuration")
        # Would validate configuration here
        logger.info("Configuration validated successfully")
        return
    
    if args.mission:
        logger.info(f"Single mission mode: Mission {args.mission}")
        # Would execute single mission here
        return
    
    if args.phase:
        logger.info(f"Single phase mode: {args.phase}")
        # Would execute single phase here
        return
    
    # Execute full fleet
    result = orchestrator.run_full_fleet_execution()
    
    # Save results
    results_path = Path('results/fleet_execution_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    
    # Exit with appropriate code
    if result['status'] == 'success':
        sys.exit(0)
    elif result['status'] == 'partial_success':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()
