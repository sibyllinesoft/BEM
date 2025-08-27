# BEM Fleet Operational Manual

## 🚀 Fleet Orchestration Guide

This manual provides comprehensive instructions for operating the BEM Fleet multi-mission research system, covering everything from initial setup through production deployment and troubleshooting.

## 📋 Prerequisites

### System Requirements

#### Hardware Requirements
```yaml
minimum_requirements:
  gpus: "2x RTX 3090 Ti (24GB each)"
  system_memory: "64GB RAM"
  storage: "1TB NVMe SSD"
  network: "High-speed internet (≥100 Mbps)"

recommended_requirements:
  gpus: "2x H100 (80GB) + 3x RTX 4090 (24GB)"
  system_memory: "128GB RAM"
  storage: "2TB NVMe SSD + 10TB data storage"
  network: "Gigabit ethernet or faster"
```

#### Software Dependencies
```bash
# Core dependencies
python >= 3.9
pytorch >= 2.1.0
cuda >= 11.8
cudnn >= 8.0

# Python packages
pip install torch transformers peft numpy scipy pandas
pip install pyyaml wandb matplotlib seaborn rich
pip install statsmodels scikit-learn psutil
```

## 🎯 Fleet Initialization

### 1. Environment Setup

#### Clone and Setup Repository
```bash
# Clone the repository
git clone <repository_url>
cd bem_fleet

# Create virtual environment
python -m venv bem_fleet_env
source bem_fleet_env/bin/activate  # Linux/Mac
# OR
bem_fleet_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

#### Environment Variables
```bash
# Set environment variables
export BEM_FLEET_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"  # Adjust for your GPU setup
export WANDB_PROJECT="bem_fleet_research"
export PYTHONPATH="${BEM_FLEET_ROOT}:${PYTHONPATH}"
```

### 2. Configuration Validation

#### Check System Configuration
```bash
# Validate hardware requirements
python scripts/validate_system_requirements.py

# Check configuration files
python scripts/validate_configuration.py --config bem_fleet_architecture.yml

# Test GPU allocation
python scripts/test_gpu_allocation.py
```

#### Configuration File Structure
```
configs/
├── bem_fleet_architecture.yml    # Main fleet configuration
├── fleet_monitoring.yaml         # Monitoring configuration
└── gates_bem2.yaml              # Acceptance gates

experiments/
├── mission_a_agentic_planner.yml # Mission A configuration
├── mission_b_living_model.yml    # Mission B configuration
├── mission_c_alignment_enforcer.yml # Mission C configuration
├── mission_d_sep.yml             # Mission D configuration
└── mission_e_long_memory.yml     # Mission E configuration
```

## 🎪 Mission Execution

### 1. Individual Mission Execution

#### Mission A: Agentic Planner
```bash
# Train behavioral cloning baseline (AR0)
python bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml \
  --phase behavioral_cloning \
  --output logs/mission_a_ar0

# Train policy gradient (AR1) 
python bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml \
  --phase policy_gradient \
  --checkpoint logs/mission_a_ar0/best_model.pt \
  --output logs/mission_a_ar1

# Evaluate mission
python bem2/evaluation/evaluation_framework.py \
  --mission mission_a \
  --model logs/mission_a_ar1/final_model.pt \
  --output results/mission_a
```

#### Mission B: Living Model
```bash
# Setup shadow mode training
python bem2/online/run_stream.py \
  --config experiments/mission_b_living_model.yml \
  --mode shadow \
  --base_model bem_v13_anchor \
  --output logs/mission_b_shadow

# Deploy online learner
python bem2/online/online_learner.py \
  --config experiments/mission_b_living_model.yml \
  --checkpoint logs/mission_b_shadow/shadow_model.pt \
  --mode online \
  --output logs/mission_b_online
```

#### Mission C: Alignment Enforcer
```bash
# Train constitutional constraints
python bem2/safety/training.py \
  --config experiments/mission_c_alignment_enforcer.yml \
  --constraint_data data/safety_training.jsonl \
  --output logs/mission_c_training

# Evaluate safety performance
python bem2/safety/evaluation.py \
  --model logs/mission_c_training/safety_model.pt \
  --test_suite eval/suites/red_team.yml \
  --output results/mission_c_safety
```

#### Mission D: SEP
```bash
# Run pretraining with scramble equivariance
python bem2/perftrack/training.py \
  --config experiments/mission_d_sep.yml \
  --data data/pretrain_corpus.jsonl \
  --variants pt1,pt2,pt3,pt4 \
  --output logs/mission_d_sep

# Evaluate OOD transfer
python bem2/perftrack/evaluation.py \
  --models logs/mission_d_sep/ \
  --ood_suite eval/suites/ood_transfer.yml \
  --output results/mission_d_transfer
```

#### Mission E: Long-Memory + SSM↔BEM
```bash
# Train memory coupling
python bem2/multimodal/training.py \
  --config experiments/mission_e_long_memory.yml \
  --context_lengths 128k,256k,512k \
  --output logs/mission_e_memory

# Evaluate long-context performance  
python bem2/multimodal/evaluation.py \
  --model logs/mission_e_memory/memory_model.pt \
  --long_context_suite eval/suites/long_context.yml \
  --output results/mission_e_long_context
```

### 2. Fleet Orchestration

#### Full Fleet Execution
```bash
# Run all missions in parallel
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --config bem_fleet_architecture.yml \
  --monitor \
  --output fleet_execution_logs

# Run specific mission subset
python scripts/orchestrate_bem_fleet.py \
  --missions mission_a,mission_b,mission_c \
  --config bem_fleet_architecture.yml \
  --monitor
```

#### Advanced Fleet Options
```bash
# Custom resource allocation
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --gpu-allocation "mission_a:0,1 mission_b:2 mission_c:3 mission_d:4 mission_e:1,4" \
  --memory-limits "mission_a:60GB mission_b:20GB mission_c:20GB mission_d:20GB mission_e:60GB"

# Staged execution with dependencies
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --respect-dependencies \
  --max-parallel 3 \
  --checkpoint-interval 3600  # Checkpoint every hour
```

## 📊 Monitoring and Dashboards

### 1. Real-Time Monitoring

#### Launch Fleet Dashboard
```bash
# Start monitoring dashboard
python monitoring/fleet_dashboard.py \
  --config configs/fleet_monitoring.yaml \
  --port 8080 \
  --refresh-interval 30

# Access dashboard at http://localhost:8080
```

#### Dashboard Features
- **Fleet Overview**: Mission status, resource utilization, system health
- **Mission Details**: Training curves, evaluation metrics, statistical significance
- **Integration Status**: Cross-mission compatibility, integration tests
- **Alerts**: Performance anomalies, resource conflicts, failures

#### Command Line Monitoring
```bash
# Check fleet status
python scripts/fleet_status.py --summary

# Monitor specific mission
python scripts/fleet_status.py --mission mission_a --detailed

# Resource utilization
python scripts/resource_monitor.py --realtime

# Log analysis
python scripts/analyze_logs.py --mission mission_b --last 24h
```

### 2. Performance Metrics

#### Key Metrics to Monitor

```yaml
mission_metrics:
  mission_a:
    primary: ["em_f1", "plan_length", "kv_hit_ratio"]
    secondary: ["latency_p50", "latency_p95", "flip_rate"]
    
  mission_b:
    primary: ["correction_speed", "aggregate_improvement", "rollback_count"]
    secondary: ["drift_rate", "canary_failures", "memory_usage"]
    
  mission_c:
    primary: ["violation_reduction", "em_f1_drop", "orthogonality"]
    secondary: ["latency_overhead", "constraint_coverage"]
    
  mission_d:
    primary: ["ood_transfer", "context_robustness", "surface_dependence"]
    secondary: ["bleu_loss", "chrf_loss", "rrs_improvement"]
    
  mission_e:
    primary: ["long_context_performance", "memory_efficiency"]
    secondary: ["latency_overhead", "perplexity_spike"]

system_metrics:
  resource_utilization: ["gpu_memory", "gpu_compute", "system_memory", "disk_io"]
  performance: ["throughput", "latency", "error_rate", "availability"]
  integration: ["compatibility_score", "interference_level", "synchronization"]
```

## 🔧 Troubleshooting Guide

### 1. Common Issues

#### GPU Memory Issues
```bash
# Symptoms: CUDA out of memory errors
# Solution 1: Reduce batch size
python scripts/adjust_batch_sizes.py --mission mission_a --reduce 50%

# Solution 2: Enable gradient checkpointing
export BEM_GRADIENT_CHECKPOINTING=true

# Solution 3: Use smaller model variants
python scripts/use_model_variant.py --mission mission_a --variant small

# Check memory usage
python scripts/gpu_memory_report.py
```

#### Training Convergence Issues
```bash
# Symptoms: Loss not decreasing, unstable training
# Diagnosis
python scripts/diagnose_training.py --logs logs/mission_a/ --detailed

# Solution: Adjust learning rates
python scripts/adjust_hyperparams.py \
  --mission mission_a \
  --learning-rate 1e-5 \
  --warmup-steps 2000

# Solution: Check data quality
python scripts/validate_data.py --dataset data/comp_tasks.jsonl
```

#### Mission Integration Failures
```bash
# Symptoms: Cross-mission compatibility issues
# Diagnosis
python scripts/diagnose_integration.py \
  --missions mission_a,mission_b \
  --check compatibility,resource_conflicts,data_flow

# Solution: Restart with isolated execution
python scripts/orchestrate_bem_fleet.py \
  --missions mission_a,mission_b \
  --isolation-mode \
  --debug
```

#### Statistical Validation Failures
```bash
# Symptoms: Statistical tests failing, significance not reached
# Check statistical power
python analysis/check_statistical_power.py \
  --mission mission_a \
  --effect-size 0.5 \
  --power 0.8

# Increase sample sizes
python scripts/increase_evaluation_samples.py \
  --mission mission_a \
  --samples 5000
```

### 2. Debug Mode Operations

#### Enable Debug Logging
```bash
# Set debug environment
export BEM_FLEET_DEBUG=true
export BEM_FLEET_LOG_LEVEL=DEBUG

# Run with detailed logging
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --debug \
  --log-file debug_fleet.log
```

#### Performance Profiling
```bash
# Profile mission execution
python -m cProfile -o mission_a_profile.prof \
  bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml

# Analyze profile
python scripts/analyze_profile.py --profile mission_a_profile.prof

# Memory profiling
python scripts/memory_profiler.py --mission mission_a --duration 3600
```

### 3. Recovery Procedures

#### Mission Failure Recovery
```bash
# Checkpoint-based recovery
python scripts/recover_mission.py \
  --mission mission_a \
  --checkpoint logs/mission_a/checkpoint_latest.pt \
  --resume

# Data corruption recovery
python scripts/repair_mission_data.py \
  --mission mission_a \
  --backup data/backups/mission_a_backup.tar.gz
```

#### System-Level Recovery
```bash
# Fleet restart with state preservation
python scripts/restart_fleet.py \
  --preserve-state \
  --config bem_fleet_architecture.yml

# Emergency shutdown
python scripts/emergency_shutdown.py --force --cleanup
```

## 📈 Statistical Analysis Operations

### 1. Statistical Validation

#### Run Statistical Pipeline
```bash
# Complete statistical analysis
python analysis/statistical_pipeline.py \
  --results results/ \
  --output analysis/statistical_results.json

# BCa Bootstrap analysis
python analysis/bca_bootstrap.py \
  --mission mission_a \
  --baseline baseline_results.json \
  --samples 10000

# FDR correction
python analysis/fdr_correction.py \
  --p-values analysis/raw_p_values.json \
  --alpha 0.05 \
  --method benjamini_hochberg
```

#### Promotion Decisions
```bash
# Check promotion gates
python analysis/check_promotion_gates.py \
  --mission mission_a \
  --gates configs/gates_bem2.yaml

# Generate promotion report
python analysis/generate_promotion_report.py \
  --all-missions \
  --output promotion_report.json
```

### 2. Performance Analysis

#### Generate Analysis Reports
```bash
# Comprehensive analysis
python analysis/comprehensive_analysis.py \
  --results results/ \
  --missions all \
  --output comprehensive_analysis_report.md

# Mission-specific analysis
python analysis/mission_analysis.py \
  --mission mission_a \
  --detailed \
  --output analysis/mission_a_detailed_report.md
```

## 🚀 Production Deployment

### 1. Pre-Deployment Validation

#### System Readiness Checklist
```bash
# Run complete validation suite
python scripts/pre_deployment_validation.py \
  --all-missions \
  --comprehensive

# Check deployment requirements
python scripts/check_deployment_readiness.py \
  --requirements production_requirements.yaml

# Security audit
python scripts/security_audit.py --comprehensive
```

#### Performance Benchmarking
```bash
# Production load testing
python scripts/load_test.py \
  --config production_load_test.yaml \
  --duration 3600 \
  --concurrent-requests 100

# Latency benchmarking
python scripts/latency_benchmark.py \
  --all-missions \
  --samples 1000 \
  --percentiles 50,95,99
```

### 2. Deployment Procedures

#### Staged Deployment
```bash
# Phase 1: Individual missions
python scripts/deploy_missions.py \
  --phase individual \
  --environment staging \
  --validate

# Phase 2: Pairwise integration
python scripts/deploy_missions.py \
  --phase pairwise \
  --environment staging \
  --integration-tests

# Phase 3: Full system
python scripts/deploy_missions.py \
  --phase full_system \
  --environment production \
  --monitoring
```

#### Container Deployment
```bash
# Build containers
docker-compose -f docker-compose.prod.yml build

# Deploy fleet
docker-compose -f docker-compose.prod.yml up -d

# Monitor deployment
docker-compose -f docker-compose.prod.yml logs -f
```

## 📊 Maintenance Operations

### 1. Regular Maintenance

#### Daily Operations
```bash
# Daily health check
python scripts/daily_health_check.py --comprehensive

# Log rotation
python scripts/rotate_logs.py --keep-days 30

# Resource cleanup
python scripts/cleanup_resources.py --automatic
```

#### Weekly Operations
```bash
# Model checkpoint cleanup
python scripts/cleanup_checkpoints.py --keep-latest 5

# Performance analysis
python scripts/weekly_performance_report.py

# Security updates
python scripts/security_update_check.py
```

### 2. Backup and Recovery

#### Data Backup
```bash
# Backup mission data
python scripts/backup_mission_data.py \
  --all-missions \
  --destination /backup/bem_fleet/$(date +%Y%m%d)

# Backup models
python scripts/backup_models.py \
  --models logs/*/final_model.pt \
  --destination /backup/models/
```

#### Configuration Management
```bash
# Version control for configs
git add configs/ experiments/
git commit -m "Configuration update for production"
git tag -a v1.3.production -m "Production configuration v1.3"
```

## 🔐 Security Operations

### 1. Security Monitoring

#### Continuous Security Checks
```bash
# Security health check
python scripts/security_health_check.py --comprehensive

# Vulnerability scanning
python scripts/vulnerability_scan.py --all-components

# Access audit
python scripts/audit_access.py --time-range 24h
```

### 2. Incident Response

#### Security Incident Response
```bash
# Emergency security shutdown
python scripts/emergency_security_shutdown.py

# Incident analysis
python scripts/security_incident_analysis.py \
  --logs security_logs/ \
  --incident-id INCIDENT_001

# Recovery procedures
python scripts/security_recovery.py \
  --incident-id INCIDENT_001 \
  --recovery-plan security_recovery_plan.yaml
```

## 📞 Support and Escalation

### 1. Support Procedures

#### Getting Help
1. **Documentation**: Check relevant sections of this manual
2. **Logs**: Review mission and system logs for error details
3. **Diagnostics**: Run diagnostic scripts for specific issues
4. **GitHub Issues**: Create detailed issue with reproduction steps
5. **Emergency Contact**: Direct contact for critical production issues

#### Common Support Requests
- **Performance Issues**: Resource allocation, optimization guidance
- **Training Problems**: Hyperparameter tuning, data quality issues
- **Integration Errors**: Cross-mission compatibility problems
- **Statistical Questions**: Validation procedures, significance testing
- **Deployment Issues**: Production setup, configuration problems

### 2. Escalation Matrix

```yaml
escalation_levels:
  level_1_user_issues:
    response_time: "4 hours"
    contacts: ["support-team"]
    examples: ["configuration", "usage questions", "minor bugs"]
    
  level_2_system_issues:
    response_time: "2 hours"
    contacts: ["engineering-team", "support-team"]
    examples: ["performance problems", "integration failures", "data corruption"]
    
  level_3_critical_issues:
    response_time: "30 minutes"
    contacts: ["on-call-engineer", "team-lead", "management"]
    examples: ["security breaches", "system outages", "data loss"]
    
  level_4_emergency:
    response_time: "immediate"
    contacts: ["all-hands", "executive-team"]
    examples: ["production down", "security compromise", "legal issues"]
```

---

This operational manual provides comprehensive guidance for successfully running the BEM Fleet system from initial setup through production deployment and ongoing maintenance. Follow these procedures to ensure reliable, secure, and efficient operation of your multi-mission research platform.