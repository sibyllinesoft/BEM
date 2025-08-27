#!/bin/bash
# BEM Reproduction Script - NeurIPS 2025 Submission
# Single GPU Complete Reproduction (RTX 3090 Ti)
# Estimated Runtime: 24-48 hours

set -e  # Exit on any error

echo "======================================================================"
echo "BEM Research Complete Reproduction Pipeline"
echo "Paper: Bolt-on Expert Modules for Retrieval-Aware Dynamic Adaptation"
echo "Venue: NeurIPS 2025 Submission"
echo "======================================================================"

# Configuration
SEEDS=5
OUTPUT_DIR="logs"
ANALYSIS_DIR="analysis"
PAPER_DIR="paper"

# Check system requirements
echo "Checking system requirements..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')" 2>/dev/null || {
    echo "ERROR: PyTorch not installed or CUDA not available"
    echo "Please install PyTorch with CUDA support"
    exit 1
}

# Environment setup
echo "Setting up environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify critical dependencies
echo "Verifying BEM implementation..."
python -c "import bem; print('✓ BEM package loaded successfully')" || {
    echo "ERROR: BEM package not found. Check installation."
    exit 1
}

echo "Verifying data integrity..."
if [ ! -f "data/train.jsonl" ] || [ ! -f "data/val.jsonl" ] || [ ! -f "data/test.jsonl" ]; then
    echo "Preparing datasets..."
    python scripts/prepare_data.py
fi

echo "Verifying FAISS indices..."
if [ ! -f "indices/domain.faiss" ]; then
    echo "Building FAISS indices..."
    python scripts/build_faiss.py --index indices/domain.faiss
fi

echo "✓ Environment setup complete"
echo ""

# Baseline Experiments
echo "======================================================================"
echo "PHASE 1: Baseline Experiments (Static LoRA)"
echo "Expected runtime: 8-10 hours"
echo "======================================================================"

if [ ! -f "$OUTPUT_DIR/L0/campaign_summary.json" ]; then
    echo "Running L0 (Static LoRA) baseline experiments..."
    python scripts/run_batch_experiments.py \
        --config experiments/L0_static_lora.yaml \
        --seeds $SEEDS \
        --output $OUTPUT_DIR/L0/ \
        --verbose
    echo "✓ L0 baseline experiments complete"
else
    echo "✓ L0 baseline results found, skipping..."
fi

# BEM Experiments
echo ""
echo "======================================================================"
echo "PHASE 2: BEM v1.1-stable Experiments" 
echo "Expected runtime: 16-20 hours"
echo "======================================================================"

if [ ! -f "$OUTPUT_DIR/B1/campaign_summary.json" ]; then
    echo "Running B1 (BEM v1.1-stable) experiments..."
    python scripts/run_batch_experiments.py \
        --config experiments/B1_bem_v11_stable.yaml \
        --seeds $SEEDS \
        --output $OUTPUT_DIR/B1/ \
        --verbose
    echo "✓ B1 BEM experiments complete"
else
    echo "✓ B1 BEM results found, skipping..."
fi

# Variant Experiments (Optional - for supplement)
echo ""
echo "======================================================================"
echo "PHASE 3: BEM Variant Experiments (Optional)"
echo "Expected runtime: 10-16 hours per variant"
echo "======================================================================"

# V2: Dual Path BEM
if [ ! -f "$OUTPUT_DIR/V2/campaign_summary.json" ]; then
    echo "Running V2 (Dual Path BEM) experiments..."
    python scripts/run_batch_experiments.py \
        --config experiments/V2_dual_path.yaml \
        --seeds $SEEDS \
        --output $OUTPUT_DIR/V2/ \
        --verbose
    echo "✓ V2 variant experiments complete"
else
    echo "✓ V2 variant results found, skipping..."
fi

# V7: FiLM Lite BEM  
if [ ! -f "$OUTPUT_DIR/V7/campaign_summary.json" ]; then
    echo "Running V7 (FiLM Lite BEM) experiments..."
    python scripts/run_batch_experiments.py \
        --config experiments/V7_film_lite.yaml \
        --seeds $SEEDS \
        --output $OUTPUT_DIR/V7/ \
        --verbose
    echo "✓ V7 variant experiments complete"
else
    echo "✓ V7 variant results found, skipping..."
fi

# V11: Learned Cache Policy
if [ ! -f "$OUTPUT_DIR/V11/campaign_summary.json" ]; then
    echo "Running V11 (Learned Cache Policy) experiments..."
    python scripts/run_batch_experiments.py \
        --config experiments/V11_learned_cache_policy.yaml \
        --seeds $SEEDS \
        --output $OUTPUT_DIR/V11/ \
        --verbose  
    echo "✓ V11 variant experiments complete"
else
    echo "✓ V11 variant results found, skipping..."
fi

# Statistical Analysis
echo ""
echo "======================================================================"
echo "PHASE 4: Statistical Analysis and Validation"
echo "Expected runtime: 1-2 hours"
echo "======================================================================"

echo "Running rigorous statistical analysis..."
python scripts/run_statistical_pipeline.py \
    --baseline $OUTPUT_DIR/L0/ \
    --treatment $OUTPUT_DIR/B1/ \
    --bootstrap-samples 10000 \
    --confidence-level 0.95 \
    --multiple-correction fdr \
    --output $ANALYSIS_DIR/statistical_results.json \
    --verbose

echo "Validating pre-registered claims..."
python scripts/init_claims.py --validate \
    --claims paper/claims.yaml \
    --results $ANALYSIS_DIR/statistical_results.json \
    --output $ANALYSIS_DIR/claim_validation.json

echo "Generating hero tables..."
python scripts/generate_tables.py \
    --results $ANALYSIS_DIR/statistical_results.json \
    --output $ANALYSIS_DIR/hero_table.csv

echo "✓ Statistical analysis complete"

# Paper Generation
echo ""
echo "======================================================================"
echo "PHASE 5: Paper Assembly and Validation"
echo "Expected runtime: 15-30 minutes"
echo "======================================================================"

echo "Generating figures..."
python scripts/generate_figures.py \
    --results $ANALYSIS_DIR/ \
    --output $PAPER_DIR/figures/

echo "Auto-generating paper sections..."
python scripts/generate_sections.py \
    --results $ANALYSIS_DIR/statistical_results.json \
    --claims $ANALYSIS_DIR/claim_validation.json \
    --output $PAPER_DIR/auto/

echo "Assembling final paper..."
python scripts/assemble_paper.py \
    --main-tex $PAPER_DIR/main.tex \
    --supplement-tex $PAPER_DIR/supplement.tex \
    --output $PAPER_DIR/

echo "Building PDFs..."
cd $PAPER_DIR
pdflatex main.tex
bibtex main
pdflatex main.tex  
pdflatex main.tex

pdflatex supplement.tex
bibtex supplement
pdflatex supplement.tex
pdflatex supplement.tex
cd ..

echo "✓ Paper generation complete"

# Final Validation
echo ""
echo "======================================================================"
echo "PHASE 6: Reproduction Validation"
echo "======================================================================"

echo "Validating reproduction results..."

# Check key results
EXPECTED_EM_IMPROVEMENT="11.1"
EXPECTED_BLEU_IMPROVEMENT="35.3"

ACTUAL_RESULTS=$(python -c "
import json
with open('$ANALYSIS_DIR/statistical_results.json') as f:
    data = json.load(f)
    
em_improvement = data['comparisons'][0]['relative_improvement_pct']
bleu_improvement = data['comparisons'][2]['relative_improvement_pct']

print(f'EM: {em_improvement:.1f}%, BLEU: {bleu_improvement:.1f}%')
")

echo "Key Results Validation:"
echo "  Expected: EM +11.1%, BLEU +35.3%"  
echo "  Actual: $ACTUAL_RESULTS"

# Check files exist
REQUIRED_FILES=(
    "$PAPER_DIR/main.pdf"
    "$PAPER_DIR/supplement.pdf" 
    "$ANALYSIS_DIR/statistical_results.json"
    "$OUTPUT_DIR/L0/campaign_summary.json"
    "$OUTPUT_DIR/B1/campaign_summary.json"
)

echo ""
echo "File Validation:"
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
    fi
done

echo ""
echo "======================================================================"
echo "REPRODUCTION COMPLETE!"
echo "======================================================================"
echo ""
echo "Generated Files:"
echo "  📄 Main Paper: $PAPER_DIR/main.pdf"
echo "  📋 Supplement: $PAPER_DIR/supplement.pdf"
echo "  📊 Statistical Results: $ANALYSIS_DIR/statistical_results.json"
echo "  📁 Complete Logs: $OUTPUT_DIR/"
echo ""
echo "Key Findings:"
python -c "
import json
with open('$ANALYSIS_DIR/statistical_results.json') as f:
    data = json.load(f)

print('  • All 4 primary metrics significantly improved:')
for comp in data['comparisons'][:4]:
    print(f'    - {comp[\"metric\"]}: +{comp[\"relative_improvement_pct\"]:.1f}% (CI: [{comp[\"ci_lower\"]:.1f}%, {comp[\"ci_upper\"]:.1f}%])')

print('')
gates_passed = sum(1 for gate in data['quality_gates'] if gate['passed'])
total_gates = len(data['quality_gates'])
print(f'  • Quality Gates: {gates_passed}/{total_gates} passed')

if gates_passed == total_gates:
    print('    ✓ All quality criteria met')
else:
    print('    ⚠ Some quality gates failed (see detailed analysis)')
"

echo ""
echo "Next Steps:"
echo "  1. Review paper/main.pdf for main results"
echo "  2. Check paper/supplement.pdf for detailed analysis" 
echo "  3. Examine analysis/statistical_results.json for raw statistics"
echo "  4. Use logs/ directory for detailed experimental traces"
echo ""
echo "Questions? See paper/repro_manifest.json for detailed instructions."
echo "======================================================================"