#!/bin/bash
# BEM Release Preparation - Restructure Documentation
# Consolidates and reorganizes documentation for professional presentation

echo "📚 Restructuring documentation..."

# Create archive directory for historical documents
mkdir -p archive/docs/
mkdir -p archive/migration/

# Move redundant/historical documentation to archive
echo "Archiving historical documentation..."
mv UNIFICATION_COMPLETE.md archive/migration/
mv MIGRATION_GUIDE.md archive/migration/
mv NAVIGATION_GUIDE.md archive/ # Keep this accessible but out of root

# Consolidate overlapping documentation
echo "Consolidating documentation..."

# Move specific architecture docs to main architecture folder
if [ ! -d "docs/architecture" ]; then
    mkdir -p docs/architecture/
fi

# Clean up redundant README files in docs/
echo "Cleaning up redundant documentation..."

# Create a single comprehensive docs index
cat > docs/README.md << 'EOF'
# BEM Documentation Hub

## 🚀 Getting Started
- [Quick Start Guide](QUICK_START.md) - 5-minute setup
- [User Guide](guides/USER_GUIDE.md) - Complete user documentation
- [Developer Guide](guides/DEVELOPER_GUIDE.md) - Development setup and contribution

## 🏗️ Architecture & Design
- [Technical Architecture](architecture/TECHNICAL_ARCHITECTURE.md) - System design and components
- [Research Methodology](RESEARCH_METHODOLOGY.md) - Scientific approach and validation
- [Statistical Framework](STATISTICAL_FRAMEWORK.md) - Statistical analysis methods

## 🔧 Operations & Deployment
- [Deployment Guide](guides/DEPLOYMENT_GUIDE.md) - Production deployment procedures
- [Operational Manual](OPERATIONAL_MANUAL.md) - System operations and monitoring
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions

## 🔬 Research & Experiments
- [Mission Specifications](missions/README.md) - Individual research missions
- [Integration Guide](INTEGRATION_GUIDE.md) - API and integration patterns
- [Monitoring Guide](MONITORING_GUIDE.md) - System monitoring and metrics

## 📋 Reference
- [Documentation Index](DOCUMENTATION_INDEX.md) - Complete documentation catalog
EOF

echo "✅ Documentation restructuring completed!"