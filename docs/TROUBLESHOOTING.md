# BEM Fleet Troubleshooting Guide

## 🚨 Emergency Procedures

### Critical System Failures

#### Complete System Shutdown
```bash
# Emergency shutdown with state preservation
python scripts/emergency_shutdown.py --preserve-state --force

# Force shutdown without state preservation (last resort)
python scripts/emergency_shutdown.py --force --no-preserve
```

#### Security Incident Response
```bash
# Immediate security lockdown
python scripts/security_lockdown.py --immediate

# Security incident analysis
python scripts/security_incident_analysis.py --incident-id $(date +%Y%m%d_%H%M%S)
```

#### Data Corruption Recovery
```bash
# Emergency data backup
python scripts/emergency_backup.py --all-missions --timestamp $(date +%Y%m%d_%H%M%S)

# Restore from latest backup
python scripts/restore_from_backup.py --latest --verify-integrity
```

## 🔧 Common Issues and Solutions

### 1. GPU Memory Issues

#### Symptoms
- CUDA out of memory errors
- System freezing during training
- Reduced batch sizes or model failures

#### Diagnosis
```bash
# Check GPU memory usage
nvidia-smi

# Detailed memory analysis
python scripts/gpu_memory_analysis.py --detailed

# Mission-specific memory profiling
python scripts/memory_profiler.py --mission mission_a --duration 300
```

#### Solutions

##### Solution 1: Reduce Batch Sizes
```bash
# Automatic batch size reduction
python scripts/adjust_batch_sizes.py --mission mission_a --reduce 50%

# Manual configuration
export BEM_BATCH_SIZE_MISSION_A=16
export BEM_BATCH_SIZE_MISSION_B=8
```

##### Solution 2: Enable Memory Optimization
```bash
# Enable gradient checkpointing
export BEM_GRADIENT_CHECKPOINTING=true

# Enable CPU offloading
export BEM_CPU_OFFLOAD=true

# Use mixed precision training
export BEM_MIXED_PRECISION=fp16
```

##### Solution 3: Resource Reallocation
```bash
# Reallocate GPU resources
python scripts/reallocate_resources.py \
  --mission mission_a \
  --primary-gpu 1 \
  --reduce-batch-size 25%
```

### 2. Training Convergence Issues

#### Symptoms
- Loss not decreasing after many epochs
- Unstable training curves
- NaN or infinite losses
- Poor evaluation metrics

#### Diagnosis
```bash
# Comprehensive training diagnostics
python scripts/diagnose_training.py --logs logs/mission_a/ --detailed

# Gradient analysis
python scripts/analyze_gradients.py --mission mission_a --window 1000

# Data quality check
python scripts/validate_training_data.py --mission mission_a
```

#### Solutions

##### Solution 1: Learning Rate Adjustment
```bash
# Automatic learning rate tuning
python scripts/tune_learning_rate.py --mission mission_a --method lr_finder

# Manual adjustment
python scripts/adjust_hyperparams.py \
  --mission mission_a \
  --learning-rate 1e-5 \
  --warmup-steps 2000 \
  --scheduler cosine
```

##### Solution 2: Data Quality Issues
```bash
# Data cleaning
python scripts/clean_training_data.py --mission mission_a --remove-outliers

# Data augmentation
python scripts/augment_training_data.py --mission mission_a --techniques all

# Data balancing
python scripts/balance_dataset.py --mission mission_a --strategy oversample
```

##### Solution 3: Model Architecture Issues
```bash
# Simplify model architecture
python scripts/simplify_architecture.py --mission mission_a --reduction 25%

# Add regularization
python scripts/add_regularization.py \
  --mission mission_a \
  --dropout 0.1 \
  --weight-decay 0.01
```

### 3. Integration Failures

#### Symptoms
- Cross-mission compatibility errors
- Resource allocation conflicts
- Data flow interruptions
- Performance degradation in integrated mode

#### Diagnosis
```bash
# Integration diagnostics
python scripts/diagnose_integration.py \
  --missions mission_a,mission_b \
  --check all

# Resource conflict analysis
python scripts/analyze_resource_conflicts.py --detailed

# Data flow validation
python scripts/validate_data_flow.py --integration router_online
```

#### Solutions

##### Solution 1: Isolation Mode
```bash
# Run missions in isolation
python scripts/orchestrate_bem_fleet.py \
  --missions mission_a,mission_b \
  --isolation-mode \
  --no-integration
```

##### Solution 2: Resource Conflict Resolution
```bash
# Automatic resource conflict resolution
python scripts/resolve_resource_conflicts.py --auto

# Manual resource reallocation
python scripts/manual_resource_allocation.py \
  --config configs/manual_allocation.yaml
```

##### Solution 3: Integration Rollback
```bash
# Rollback to working integration state
python scripts/rollback_integration.py --to-checkpoint latest_stable

# Reset integration configuration
python scripts/reset_integration.py --missions mission_a,mission_b
```

### 4. Statistical Validation Failures

#### Symptoms
- Statistical tests not reaching significance
- Bootstrap confidence intervals too wide
- FDR correction rejecting too many tests
- Power analysis showing insufficient samples

#### Diagnosis
```bash
# Statistical diagnostics
python analysis/diagnose_statistics.py --mission mission_a --detailed

# Power analysis
python analysis/power_analysis.py --mission mission_a --effect-size 0.5

# Sample size analysis
python analysis/sample_size_analysis.py --all-missions
```

#### Solutions

##### Solution 1: Increase Sample Sizes
```bash
# Increase evaluation samples
python scripts/increase_evaluation_samples.py \
  --mission mission_a \
  --samples 5000

# Add more random seeds
python scripts/add_random_seeds.py --seeds 6,7,8,9,10
```

##### Solution 2: Effect Size Analysis
```bash
# Check if effects are practically significant
python analysis/effect_size_analysis.py --mission mission_a

# Adjust minimum effect size thresholds
python scripts/adjust_effect_size_thresholds.py \
  --mission mission_a \
  --min-effect-size 0.3
```

##### Solution 3: Statistical Method Adjustment
```bash
# Use non-parametric tests
python analysis/run_nonparametric_tests.py --mission mission_a

# Adjust FDR level
python analysis/adjust_fdr_level.py --alpha 0.1
```

### 5. Performance Degradation

#### Symptoms
- Slower than expected inference times
- High memory usage
- Poor throughput
- System responsiveness issues

#### Diagnosis
```bash
# Performance profiling
python scripts/performance_profiler.py --mission mission_a --duration 300

# Bottleneck analysis
python scripts/analyze_bottlenecks.py --system-wide

# Resource utilization analysis
python scripts/analyze_utilization.py --detailed
```

#### Solutions

##### Solution 1: Model Optimization
```bash
# Model quantization
python scripts/quantize_models.py --mission mission_a --precision int8

# Model pruning
python scripts/prune_models.py --mission mission_a --sparsity 0.2

# Kernel optimization
python scripts/optimize_kernels.py --mission mission_a
```

##### Solution 2: Caching Optimization
```bash
# Enable aggressive caching
python scripts/optimize_caching.py --mission mission_a --aggressive

# Cache warming
python scripts/warm_cache.py --all-missions

# Cache size tuning
python scripts/tune_cache_sizes.py --auto
```

##### Solution 3: Parallelization
```bash
# Enable model parallelism
export BEM_MODEL_PARALLEL=true
export BEM_PARALLEL_DEVICES="0,1"

# Optimize data loading
python scripts/optimize_data_loading.py --workers 8 --prefetch 2
```

## 🔍 Debugging Procedures

### Debug Mode Activation

#### Enable System-Wide Debugging
```bash
# Set debug environment variables
export BEM_FLEET_DEBUG=true
export BEM_FLEET_LOG_LEVEL=DEBUG
export BEM_FLEET_PROFILE=true

# Run with detailed logging
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --debug \
  --log-file debug_$(date +%Y%m%d_%H%M%S).log \
  --profile
```

#### Mission-Specific Debugging
```bash
# Debug specific mission
python bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml \
  --debug \
  --verbose \
  --save-intermediates
```

### Log Analysis Tools

#### Automated Log Analysis
```bash
# Analyze error patterns
python scripts/analyze_error_patterns.py --logs logs/ --time-range 24h

# Performance log analysis
python scripts/analyze_performance_logs.py --mission mission_a

# Statistical analysis of logs
python scripts/statistical_log_analysis.py --pattern "training_loss"
```

#### Interactive Log Exploration
```bash
# Real-time log monitoring
python scripts/monitor_logs.py --mission mission_a --follow

# Log search and filtering
python scripts/search_logs.py \
  --pattern "CUDA out of memory" \
  --time-range "2024-01-01 to 2024-01-02"
```

### Profiling and Performance Analysis

#### CPU Profiling
```bash
# Profile Python execution
python -m cProfile -o mission_a_profile.prof \
  bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml

# Analyze profile results
python scripts/analyze_cpu_profile.py --profile mission_a_profile.prof
```

#### GPU Profiling
```bash
# NVIDIA profiling
nsys profile -t cuda,nvtx -o mission_a_gpu_profile \
  python bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml

# Analyze GPU profile
python scripts/analyze_gpu_profile.py --profile mission_a_gpu_profile.qdrep
```

#### Memory Profiling
```bash
# Memory usage profiling
python -m memory_profiler bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml

# Detailed memory analysis
python scripts/detailed_memory_analysis.py --mission mission_a
```

## 🔄 Recovery Procedures

### Mission Recovery

#### Checkpoint-Based Recovery
```bash
# List available checkpoints
python scripts/list_checkpoints.py --mission mission_a

# Recover from specific checkpoint
python scripts/recover_from_checkpoint.py \
  --mission mission_a \
  --checkpoint logs/mission_a/checkpoint_step_5000.pt \
  --validate

# Automatic recovery from latest checkpoint
python scripts/auto_recover.py --mission mission_a --latest
```

#### State Recovery
```bash
# Recover training state
python scripts/recover_training_state.py \
  --mission mission_a \
  --state-file logs/mission_a/training_state.json

# Recover optimizer state
python scripts/recover_optimizer_state.py \
  --mission mission_a \
  --optimizer-file logs/mission_a/optimizer_state.pt
```

### Data Recovery

#### Training Data Recovery
```bash
# Recover corrupted training data
python scripts/recover_training_data.py \
  --mission mission_a \
  --backup data/backups/mission_a_data_backup.tar.gz \
  --verify-integrity

# Regenerate synthetic data
python scripts/regenerate_synthetic_data.py --mission mission_a --samples 10000
```

#### Configuration Recovery
```bash
# Restore configuration from backup
python scripts/restore_configuration.py \
  --mission mission_a \
  --config-backup configs/backups/mission_a_backup.yml

# Reset to default configuration
python scripts/reset_to_defaults.py --mission mission_a
```

### System Recovery

#### Full System Recovery
```bash
# System-wide recovery procedure
python scripts/full_system_recovery.py \
  --backup-timestamp 20240101_120000 \
  --verify-integrity \
  --test-functionality

# Partial system recovery
python scripts/partial_system_recovery.py \
  --missions mission_a,mission_b \
  --preserve-other-missions
```

## 📞 Getting Help

### Diagnostic Information Collection

#### System Information
```bash
# Collect comprehensive system info
python scripts/collect_system_info.py --output system_info_$(date +%Y%m%d).json

# Hardware diagnostics
python scripts/hardware_diagnostics.py --full-test

# Environment diagnostics
python scripts/environment_diagnostics.py --check-all
```

#### Issue Report Generation
```bash
# Generate comprehensive issue report
python scripts/generate_issue_report.py \
  --mission mission_a \
  --issue-type training_failure \
  --include-logs \
  --include-configs \
  --include-data-samples

# Create minimal reproduction case
python scripts/create_reproduction_case.py \
  --issue-type statistical_validation_failure \
  --minimal
```

### Support Escalation

#### Level 1: Self-Service
1. **Check Documentation**: Review relevant sections
2. **Run Diagnostics**: Use automated diagnostic tools
3. **Check Logs**: Analyze error messages and patterns
4. **Try Common Solutions**: Apply standard fixes

#### Level 2: Community Support
1. **GitHub Issues**: Create detailed issue with reproduction steps
2. **Community Forums**: Ask questions in community discussions
3. **Documentation Feedback**: Report documentation gaps

#### Level 3: Expert Support
1. **Direct Contact**: Reach out to system experts
2. **Priority Support**: For critical production issues
3. **Custom Solutions**: For unique or complex problems

### Issue Reporting Template

```markdown
## Issue Report Template

### System Information
- **BEM Fleet Version**: 
- **Python Version**: 
- **GPU Models**: 
- **CUDA Version**: 
- **OS**: 

### Issue Description
- **Mission(s) Affected**: 
- **Issue Type**: 
- **Symptoms**: 
- **When Started**: 

### Reproduction Steps
1. 
2. 
3. 

### Error Messages
```
[Paste error messages here]
```

### Attempted Solutions
- [ ] Solution 1: Description
- [ ] Solution 2: Description

### Additional Context
[Any other relevant information]

### Attachments
- [ ] System info file
- [ ] Log files
- [ ] Configuration files
- [ ] Minimal reproduction case
```

## 🛡️ Prevention and Monitoring

### Proactive Monitoring Setup

#### Health Monitoring
```bash
# Setup continuous health monitoring
python scripts/setup_health_monitoring.py \
  --all-missions \
  --check-interval 60 \
  --alert-thresholds configs/alert_thresholds.yaml

# Setup automated alerts
python scripts/setup_automated_alerts.py \
  --email-notifications \
  --slack-integration \
  --sms-critical-only
```

#### Performance Monitoring
```bash
# Setup performance baselines
python scripts/setup_performance_baselines.py --all-missions

# Enable continuous performance monitoring
python scripts/enable_performance_monitoring.py \
  --metrics latency,memory,throughput \
  --alert-on-degradation 15%
```

### Preventive Maintenance

#### Regular Maintenance Tasks
```bash
# Daily maintenance script
cat > scripts/daily_maintenance.sh << 'EOF'
#!/bin/bash
# Daily BEM Fleet maintenance

# Health checks
python scripts/daily_health_check.py --comprehensive

# Log rotation
python scripts/rotate_logs.py --keep-days 30

# Checkpoint cleanup
python scripts/cleanup_old_checkpoints.py --keep-latest 5

# Performance check
python scripts/performance_regression_check.py --baseline-window 7d

# Security scan
python scripts/daily_security_scan.py
EOF

chmod +x scripts/daily_maintenance.sh
```

#### Scheduled Maintenance
```bash
# Setup cron job for daily maintenance
echo "0 2 * * * /path/to/bem_fleet/scripts/daily_maintenance.sh" | crontab -

# Weekly maintenance
echo "0 1 * * 0 /path/to/bem_fleet/scripts/weekly_maintenance.sh" | crontab -
```

### Early Warning Systems

#### Anomaly Detection
```bash
# Setup anomaly detection
python scripts/setup_anomaly_detection.py \
  --metrics training_loss,evaluation_score,latency \
  --sensitivity medium \
  --learning-period 7d

# Setup statistical process control
python scripts/setup_spc_monitoring.py \
  --control-charts all \
  --alert-on-out-of-control
```

This troubleshooting guide provides comprehensive coverage of common issues and their solutions, enabling efficient problem resolution and system maintenance for the BEM Fleet.