# BEM Fleet Quick Start Guide

## 🚀 Get Running in 5 Minutes

This guide will get you running the BEM Fleet multi-mission research system as quickly as possible.

## 📋 Prerequisites

### System Requirements
- **GPU**: 2x RTX 3090 Ti (24GB each) minimum
- **RAM**: 64GB system memory
- **Storage**: 1TB NVMe SSD
- **OS**: Ubuntu 20.04+ or similar Linux

### Quick Environment Check
```bash
# Check GPU availability
nvidia-smi

# Check Python version (3.9+ required)
python --version

# Check available disk space
df -h
```

## ⚡ Installation

### 1. Clone and Setup
```bash
# Clone repository
git clone <repository_url>
cd bem_fleet

# Create virtual environment
python -m venv bem_fleet_env
source bem_fleet_env/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Quick Validation
```bash
# Validate installation
python scripts/validate_installation.py

# Quick system test
python scripts/quick_system_test.py
```

### 3. Environment Configuration
```bash
# Set environment variables
export BEM_FLEET_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"  # Adjust for your GPUs
export PYTHONPATH="${BEM_FLEET_ROOT}:${PYTHONPATH}"
```

## 🎯 Quick Demo

### Option 1: Single Mission Demo (2 minutes)
```bash
# Run Mission A (Agentic Planner) demo
python scripts/demo_mission_a.py --quick --samples 100

# Expected output: Training and evaluation results
```

### Option 2: Full Fleet Demo (15 minutes)
```bash
# Run all 5 missions in parallel
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --quick-mode \
  --samples 1000
```

### Option 3: Statistical Validation Demo (5 minutes)
```bash
# Run statistical validation example
python analysis/demo_statistical_validation.py --quick
```

## 🔬 Understanding the Output

### Mission A Output Example
```
🎉 Mission A (Agentic Planner) Results:
═══════════════════════════════════════
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                  ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ EM/F1 Improvement       │ +2.3%         │
│ Plan Length (avg)       │ 2.1           │
│ Latency Overhead        │ +8.4%         │
│ Statistical Significance│ p < 0.001     │
└─────────────────────────┴───────────────┘
✅ Mission A: SUCCESS (Gates Passed)
```

### Fleet Status Dashboard
```
🚀 BEM Fleet Status Overview:
═════════════════════════════════
Mission A (Router)     │ ✅ COMPLETE │ +2.3% EM/F1
Mission B (Online)     │ 🔄 RUNNING  │ 847 prompts
Mission C (Safety)     │ ✅ COMPLETE │ -31.2% violations
Mission D (SEP)        │ 🔄 TRAINING │ Epoch 15/50
Mission E (Memory)     │ ⏸️  QUEUED   │ Pending
═════════════════════════════════
Overall Fleet Health: 🟢 HEALTHY
```

## 📊 Next Steps

### 1. Run Full Experiments
```bash
# Full 60-day research sprint simulation (4 hours)
python scripts/orchestrate_bem_fleet.py \
  --all-missions \
  --full-experiment \
  --config bem_fleet_architecture.yml
```

### 2. Launch Monitoring Dashboard
```bash
# Start real-time monitoring (separate terminal)
python monitoring/fleet_dashboard.py --port 8080

# Access dashboard at http://localhost:8080
```

### 3. Explore Individual Missions
```bash
# Mission A: Agentic Planner
python bem2/router/train_agentic_router.py \
  --config experiments/mission_a_agentic_planner.yml

# Mission B: Living Model
python bem2/online/online_learner.py \
  --config experiments/mission_b_living_model.yml

# Mission C: Alignment Enforcer
python bem2/safety/training.py \
  --config experiments/mission_c_alignment_enforcer.yml
```

## 🔧 Configuration

### Basic Configuration
```yaml
# Edit bem_fleet_architecture.yml for your setup
fleet_config:
  mission_count: 5
  gpu_allocation:
    mission_a: "cuda:0"
    mission_b: "cuda:1"
    # etc.
  quick_mode:
    enabled: true
    samples_per_mission: 1000
    epochs_per_mission: 10
```

### Resource Allocation
```bash
# Adjust for your hardware
export BEM_GPU_MEMORY_LIMIT="20GB"  # Per GPU
export BEM_MAX_PARALLEL_MISSIONS=3
export BEM_BATCH_SIZE_SCALE=0.5
```

## 🛠️ Common Quick Fixes

### GPU Memory Issues
```bash
# Reduce resource usage
export BEM_GRADIENT_CHECKPOINTING=true
export BEM_MIXED_PRECISION=fp16
export BEM_BATCH_SIZE_SCALE=0.25
```

### Slow Training
```bash
# Enable optimizations
export BEM_COMPILE_MODEL=true
export BEM_DATALOADER_WORKERS=4
export BEM_PREFETCH_FACTOR=2
```

### Installation Issues
```bash
# Clean installation
pip uninstall -r requirements.txt -y
pip install -r requirements.txt --no-cache-dir

# Update dependencies
pip install --upgrade torch transformers
```

## 📖 What's Next?

### Detailed Documentation
- **[Research Methodology](RESEARCH_METHODOLOGY.md)** - Scientific approach and validation
- **[Technical Architecture](TECHNICAL_ARCHITECTURE.md)** - System design and components  
- **[Operational Manual](OPERATIONAL_MANUAL.md)** - Complete operation procedures
- **[Mission Specifications](missions/README.md)** - Individual mission details

### Advanced Usage
- **Statistical Analysis**: Deep dive into BCa bootstrap and FDR correction
- **Performance Optimization**: GPU optimization and scaling techniques
- **Integration Patterns**: Cross-mission coordination and data flow
- **Production Deployment**: Scale to production workloads

### Getting Help
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **GitHub Issues**: Report bugs or request features
- **Community Forums**: Ask questions and share experiences

## 🎯 Success Indicators

You've successfully completed the quick start if you see:

✅ **Environment Validated**: All dependencies installed correctly  
✅ **System Test Passed**: Hardware and software compatibility confirmed  
✅ **Demo Completed**: At least one mission demo ran successfully  
✅ **Output Understood**: You can interpret the results and metrics  
✅ **Next Steps Identified**: You know which advanced topics to explore  

## ⚡ One-Command Start

For the absolute fastest start:

```bash
# Complete quick start in one command
curl -sSL https://raw.githubusercontent.com/bem-fleet/bem-fleet/main/scripts/quick_start.sh | bash
```

This script will:
1. Check system requirements
2. Install dependencies  
3. Run quick validation
4. Execute demo mission
5. Display results and next steps

---

**Estimated Total Time**: 5-15 minutes depending on download speeds and hardware  
**Next Recommended Reading**: [Research Methodology](RESEARCH_METHODOLOGY.md) or [Technical Architecture](TECHNICAL_ARCHITECTURE.md)