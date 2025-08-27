#!/bin/bash
# LaTeX Dependencies Installation Script for BEM Paper Build
# Generated on: $(date)
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🔍 Checking system requirements..."
echo "This will install LaTeX packages needed to build the BEM research paper."

echo "📦 Installing LaTeX dependencies..."
# Install essential LaTeX packages
sudo apt update
sudo apt install -y texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra biber

# Install algorithm packages specifically
sudo apt install -y texlive-science

# Install additional recommended packages
sudo apt install -y texlive-publishers texlive-latex-recommended

echo "✅ Verifying installation..."
# Check if key packages are available
if pdflatex --version > /dev/null 2>&1; then
    echo "✓ pdflatex is available"
else
    echo "✗ pdflatex not found"
    exit 1
fi

if bibtex --version > /dev/null 2>&1; then
    echo "✓ bibtex is available"
else
    echo "✗ bibtex not found"
    exit 1
fi

echo "🎉 LaTeX dependencies installation complete!"
echo ""
echo "You can now build the BEM paper with:"
echo "cd archive/paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"