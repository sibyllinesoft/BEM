#!/bin/bash
# BEM Release Preparation - Setup Git LFS for Large Files
# Configures Git LFS to handle model files and large assets

echo "📦 Setting up Git LFS for large files..."

# Initialize Git LFS if not already done
git lfs install

# Create .gitattributes for large file patterns
cat > .gitattributes << 'EOF'
# Git LFS configuration for BEM repository

# Model files
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text

# Data files
*.faiss filter=lfs diff=lfs merge=lfs -text
*.jsonl filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text

# Compressed archives
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.tar.bz2 filter=lfs diff=lfs merge=lfs -text

# Generated images (if kept in repo)
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.pdf filter=lfs diff=lfs merge=lfs -text
EOF

# Track existing large files
echo "Tracking existing large files with Git LFS..."

# Find and track model files
find models/ -name "*.safetensors" -exec git lfs track {} \;
find models/ -name "*.bin" -exec git lfs track {} \;

# Find and track data files
find data/ -name "*.faiss" -exec git lfs track {} \;
find data/ -name "*.jsonl" -exec git lfs track {} \;

echo "✅ Git LFS setup completed!"
echo "📝 Note: Run 'git add .gitattributes && git commit -m \"Add Git LFS configuration\"' to commit changes"