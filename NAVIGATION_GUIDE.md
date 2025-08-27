# 🧭 Repository Navigation Guide

## 🎯 Quick Navigation by Use Case

### 👨‍💻 **I want to understand the system**
```
📖 Start Here:
├── README.md                          # Main overview
├── docs/README.md                     # Detailed system introduction
└── docs/QUICK_START.md               # 5-minute getting started

🏗️ Architecture Deep Dive:
├── docs/architecture/TECHNICAL_ARCHITECTURE.md
├── docs/architecture/RESEARCH_METHODOLOGY_CONFIRMATION.md
└── src/bem2/README.md                # Modern implementation overview
```

### 🧪 **I want to run experiments**
```
🚀 Start Experiments:
├── scripts/demos/                     # Demo scripts for quick testing
│   ├── demo_simple_bem.py            # Basic BEM functionality
│   ├── demo_phase5_complete.py       # Full feature demonstration
│   └── demo_hierarchical_bem.py      # Advanced routing
├── scripts/utilities/run_bem_experiments.py  # Main experiment runner
└── experiments/                       # All experiment configurations

📊 Check Results:
├── results/                          # Generated outputs and analysis
└── logs/                            # Detailed experiment logs
```

### 🔬 **I want to develop/contribute**
```
💻 Core Code:
├── src/bem2/                        # Primary implementation (v2.0)
│   ├── router/                      # Agentic routing system
│   ├── online/                      # Online learning
│   ├── multimodal/                  # Vision-text integration
│   ├── safety/                      # Constitutional safety
│   └── perftrack/                   # Performance variants
└── src/bem_legacy/                  # Legacy implementation (v0.5)

🧪 Testing:
├── tests/                           # All test files
│   ├── test_*.py                    # Current implementation tests
│   └── legacy/                      # Legacy implementation tests
└── scripts/validation/              # Validation scripts

📚 Development Guides:
├── docs/guides/DEVELOPER_GUIDE.md
├── docs/guides/DEPLOYMENT_GUIDE.md
└── docs/TROUBLESHOOTING.md
```

### 📊 **I want to see research results**
```
📈 Results & Analysis:
├── results/                         # Experiment outputs
│   ├── *.json                      # Raw experimental data
│   ├── *.csv                       # Tabular results
│   ├── *.png                       # Generated figures
│   └── analysis/                   # Detailed analysis reports
├── logs/                           # Detailed execution logs
└── archive/                        # Historical results

📖 Research Documentation:
├── docs/STATISTICAL_FRAMEWORK.md   # Statistical methodology
├── docs/RESEARCH_METHODOLOGY.md    # Research approach
└── archive/phase_reports/          # Historical phase documentation
```

### 🔧 **I want to deploy/integrate**
```
🚀 Deployment:
├── deployment/                      # Production deployment files
│   ├── manifests/                  # Kubernetes/Docker configs
│   ├── workflows/                  # CI/CD workflows
│   └── monitoring/                 # Monitoring configurations
├── docs/guides/DEPLOYMENT_GUIDE.md
├── docs/OPERATIONAL_MANUAL.md
└── requirements.txt                # Dependencies

🔌 Integration:
├── docs/INTEGRATION_GUIDE.md
├── docs/API_REFERENCE.md
└── src/bem2/__init__.py            # Public API exports
```

## 🗂️ Directory Purpose Reference

| Directory | Purpose | Key Contents |
|-----------|---------|--------------|
| `src/` | Core implementations | bem2 (primary), bem_legacy (archived) |
| `docs/` | All documentation | guides, architecture, mission specs |
| `tests/` | All test files | current tests + legacy tests |
| `scripts/` | Utilities & demos | validation, demos, utilities |
| `experiments/` | Experiment configs | YAML configs for all experiments |
| `data/` | Datasets & resources | corpora, indices, templates |
| `results/` | Generated outputs | analysis, figures, raw data |
| `logs/` | Runtime logs | execution logs by experiment |
| `models/` | Trained models | base models, value models, vision |
| `archive/` | Historical files | phase reports, old documentation |
| `deployment/` | Production files | manifests, workflows, monitoring |

## 🏷️ File Naming Conventions

### Scripts
- `demo_*.py` → `scripts/demos/` - Demonstration scripts
- `validate_*.py` → `scripts/validation/` - Validation scripts  
- `test_*.py` → `tests/` - Test files
- `run_*.py` → `scripts/utilities/` - Execution utilities

### Documentation
- `*README*.md` → Primary documentation files
- `*GUIDE*.md` → User/developer guides  
- `*SUMMARY*.md` → Summary/completion reports → `archive/`
- `*STATUS*.md` → Phase status reports → `archive/phase_reports/`

### Configuration
- `*.yml`, `*.yaml` → `experiments/` - Experiment configurations
- `requirements.txt` → Root - Python dependencies
- `*.json` → `results/` - Data files and results

## 🔍 Finding Specific Features

### BEM Core Features
```
Simple BEM:           src/bem_legacy/simple_bem.py
Hierarchical BEM:     src/bem_legacy/hierarchical_bem.py  
Retrieval BEM:        src/bem_legacy/retrieval_bem.py
Multi-BEM:            src/bem_legacy/multi_bem.py
Advanced Features:    src/bem_legacy/advanced_variants.py
```

### BEM v2.0 Features
```
Agentic Router:       src/bem2/router/agentic_router.py
Online Learning:      src/bem2/online/online_learner.py
Multimodal:           src/bem2/multimodal/controller_integration.py
Safety:               src/bem2/safety/safety_controller.py
Performance:          src/bem2/perftrack/pt1_head_gating.py (etc.)
```

### Experiments & Validation
```
Experiment Runner:    scripts/utilities/run_bem_experiments.py
Demo Suite:           scripts/demos/demo_*.py
Validation Suite:     scripts/validation/validate_*.py
Statistical Analysis: results/analysis/
```

## 🕐 Historical Context

### Implementation Generations
1. **BEM v0.5** (`src/bem_legacy/`) - 5-phase comprehensive implementation
2. **BEM v2.0** (`src/bem2/`) - Mission-focused modern architecture

### Archive Organization
```
archive/
├── phase_reports/        # Original 5-phase development reports
├── experimental_reports/ # Security, VC0, advanced variants
├── scattered_docs/       # Miscellaneous documentation  
├── paper/               # Academic paper materials
└── *.md                 # Final validation and completion reports
```

## 🎯 Recommended Entry Points

### New Users
1. `README.md` - System overview
2. `docs/QUICK_START.md` - Get running quickly
3. `scripts/demos/demo_simple_bem.py` - See it working

### Developers
1. `docs/guides/DEVELOPER_GUIDE.md` - Development setup
2. `src/bem2/` - Modern codebase
3. `tests/` - Test suite

### Researchers  
1. `docs/RESEARCH_METHODOLOGY.md` - Research framework
2. `docs/STATISTICAL_FRAMEWORK.md` - Statistical approach
3. `results/` - Experimental results

### Operators
1. `docs/OPERATIONAL_MANUAL.md` - Operations guide
2. `deployment/` - Deployment configurations
3. `docs/TROUBLESHOOTING.md` - Problem resolution

---

**TIP**: Use your editor's "Go to File" feature (Ctrl+P in VSCode) to quickly navigate to specific files using the patterns above!