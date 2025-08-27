# BEM 2.0 Online Learning System (OL0)

This directory implements the Lifelong Learner (OL0) for BEM 2.0, enabling safe controller-only updates using feedback signals while preserving system stability.

## Overview

The BEM 2.0 online learning system implements the requirements from `TODO.md`:

- **Controller-only updates**: Only routing heads are updated, BEM matrices remain unchanged
- **EWC/Prox regularization** to prevent catastrophic forgetting  
- **10k sample replay buffer** for knowledge retention
- **Canary testing** before applying updates
- **Automatic rollback** on safety failures
- **Between-prompts-only** update policy
- **24-hour soak test** goal with +≥1% improvement and no canary regressions

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  StreamProcessor│────│ FeedbackProcessor│────│  OnlineLearner  │
│  (Live signals) │    │ (Convert to data)│    │ (Core learning) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CanaryGate    │────│   DriftMonitor   │────│ CheckpointMgr   │
│ (Safety tests)  │    │ (Auto rollback)  │    │ (Rollback cap.) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ EWCRegularizer  │────│   ReplayBuffer   │────│ OnlineEvaluator │
│(Prevent forget) │    │ (10k samples)    │    │ (24h soak test) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Components

### Core Learning
- **OnlineLearner**: Main orchestrator implementing the online learning loop
- **EWCRegularizer**: Elastic Weight Consolidation with diagonal Fisher information
- **ReplayBuffer**: Experience replay with prioritized sampling (10k samples)

### Safety & Monitoring  
- **CanaryGate**: Safety testing before applying updates
- **DriftMonitor**: KL divergence and parameter norm monitoring with auto-rollback
- **CheckpointManager**: Safe rollback capability

### Streaming & Processing
- **StreamProcessor**: Live feedback stream processing (thumbs, tool results)
- **FeedbackProcessor**: Convert feedback signals to training data

### Evaluation & Warmup
- **OnlineEvaluator**: 24-hour soak test and performance tracking
- **WarmupManager**: Initialize from AR1 checkpoint with Fisher computation

## Usage

### 1. Basic Streaming Mode
```bash
# Run with live feedback processing
python -m bem2.online.run_stream --config example_config.json
```

### 2. Warmup from AR1 Checkpoint
```bash
# Initialize from AR1 checkpoint (as specified in TODO.md)
python -m bem2.online.run_stream \
    --config example_config.json \
    --warmup-from /path/to/ar1_checkpoint.pt
```

### 3. 24-Hour Soak Test
```bash
# Run the full 24-hour soak test
python -m bem2.online.run_stream \
    --config example_config.json \
    --warmup-from /path/to/ar1_checkpoint.pt \
    --soak-test \
    --duration-hours 24
```

### 4. Generate Evaluation Report
```bash
# Generate performance report
python -m bem2.online.run_stream \
    --config example_config.json \
    --report-only
```

## Configuration

See `example_config.json` for a complete configuration example. Key sections:

- **online_learning**: Core learning parameters (learning rate, EWC lambda, etc.)
- **streaming**: Live feedback processing settings
- **warmup**: AR1 checkpoint initialization settings  
- **safety**: Auto-rollback and safety thresholds
- **evaluation**: Soak test and monitoring configuration

## Programmatic Usage

```python
from bem2.online import (
    OnlineLearner, OnlineLearningConfig,
    OnlineEvaluator, run_24hour_soak_test,
    WarmupManager
)

# Initialize components
config = OnlineLearningConfig(learning_rate=1e-5, ewc_lambda=1000.0)
learner = OnlineLearner(config)
evaluator = OnlineEvaluator()

# Warmup from AR1 checkpoint
warmup_manager = WarmupManager()
warmup_result = await warmup_manager.warmup_from_ar1(checkpoint, learner.model)

# Set baseline and run soak test
evaluator.set_baseline_metrics(warmup_result.baseline_metrics)
soak_result = run_24hour_soak_test(evaluator, warmup_result.baseline_metrics)

if soak_result.success:
    print(f"✅ Soak test passed: {soak_result.aggregate_improvement:+.2f}% improvement")
else:
    print(f"❌ Soak test failed: {soak_result.summary}")
```

## Safety Mechanisms

The system implements multiple safety layers:

1. **EWC Regularization**: Prevents catastrophic forgetting of old knowledge
2. **Canary Testing**: Validates updates before applying them to production
3. **Drift Monitoring**: Continuous monitoring with automatic rollback triggers
4. **Replay Buffer**: Maintains diverse experience for stable learning
5. **Controller-only Updates**: Limits changes to routing heads only
6. **Between-prompts Policy**: Only updates between conversation turns

## Monitoring & Evaluation

The system tracks comprehensive metrics:

- **Performance**: Task success rate, response quality, user satisfaction  
- **Safety**: Canary pass rate, safety violations, rollback frequency
- **Stability**: KL divergence trends, parameter norm stability
- **Learning**: Update success rate, learning efficiency

## Requirements from TODO.md

✅ **OL0 Requirements**:
- 24-hour soak test capability
- +≥1% aggregate improvement goal  
- No canary regressions requirement
- Controller-only updates (routing heads only)
- EWC/Prox regularization for stability
- 10k sample replay buffer
- Automatic rollback on failures
- Between-prompts-only updates

✅ **Implementation Complete**:
- All core components implemented
- Safety mechanisms in place
- Comprehensive evaluation system
- Streaming feedback processing
- AR1 checkpoint warmup capability

## File Structure

```
bem2/online/
├── __init__.py              # Package exports
├── interfaces.py            # Common data structures
├── online_learner.py        # Main learning orchestrator  
├── ewc_regularizer.py       # EWC with Fisher information
├── replay_buffer.py         # 10k experience replay
├── canary_gate.py           # Safety testing
├── drift_monitor.py         # Monitoring & auto-rollback
├── checkpointing.py         # Rollback capability
├── streaming.py             # Live feedback streams
├── feedback_processor.py    # Signal to data conversion
├── warmup.py                # AR1 checkpoint initialization
├── evaluation.py            # 24h soak test & metrics
├── run_stream.py            # Main runner script
├── example_config.json      # Configuration example
└── README.md                # This file
```