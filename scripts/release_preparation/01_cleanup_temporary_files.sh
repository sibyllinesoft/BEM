#!/bin/bash
# BEM Release Preparation - Cleanup Temporary Files
# This script removes all temporary, cache, and generated files

echo "🧹 Starting BEM repository cleanup..."

# Remove model cache directories (duplicated cached models)
echo "Removing model cache directories..."
rm -rf models/base/model_cache/
rm -rf models/base/tokenizer_cache/
rm -rf models/dialogpt-small/model_cache/
rm -rf models/dialogpt-small/tokenizer_cache/

# Remove backup directories
echo "Removing backup directories..."
rm -rf experiments_backup_20250827_020601/

# Remove conversion logs and temporary files
echo "Removing conversion logs and temporary files..."
rm -f conversion_log.txt
rm -f conversion_report_*.txt
rm -f *.tmp
rm -f *.temp

# Clean up generated files in results that should be regenerated
echo "Cleaning generated results files..."
rm -f results/*.tex
rm -f results/*.csv
rm -f results/*_report.md

# Remove any .pyc files and __pycache__ directories
echo "Removing Python cache files..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove any editor temporary files
echo "Removing editor temporary files..."
find . -name "*~" -delete
find . -name "*.swp" -delete
find . -name "*.swo" -delete

# Remove any OS-generated files
echo "Removing OS-generated files..."
find . -name ".DS_Store" -delete
find . -name "Thumbs.db" -delete

echo "✅ Temporary file cleanup completed!"