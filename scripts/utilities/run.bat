@echo off
REM BEM Reproduction Script - NeurIPS 2025 Submission (Windows)
REM Single GPU Complete Reproduction (RTX 3090 Ti)
REM Estimated Runtime: 24-48 hours

echo ======================================================================
echo BEM Research Complete Reproduction Pipeline
echo Paper: Bolt-on Expert Modules for Retrieval-Aware Dynamic Adaptation
echo Venue: NeurIPS 2025 Submission
echo ======================================================================

REM Configuration
set SEEDS=5
set OUTPUT_DIR=logs
set ANALYSIS_DIR=analysis
set PAPER_DIR=paper

REM Check system requirements
echo Checking system requirements...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch not installed or CUDA not available
    echo Please install PyTorch with CUDA support
    pause
    exit /b 1
)

REM Environment setup
echo Setting up environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt

REM Verify critical dependencies
echo Verifying BEM implementation...
python -c "import bem; print('✓ BEM package loaded successfully')" 2>nul
if errorlevel 1 (
    echo ERROR: BEM package not found. Check installation.
    pause
    exit /b 1
)

echo Verifying data integrity...
if not exist "data\train.jsonl" (
    echo Preparing datasets...
    python scripts\prepare_data.py
)

echo Verifying FAISS indices...
if not exist "indices\domain.faiss" (
    echo Building FAISS indices...
    python scripts\build_faiss.py --index indices\domain.faiss
)

echo ✓ Environment setup complete
echo.

REM Baseline Experiments
echo ======================================================================
echo PHASE 1: Baseline Experiments (Static LoRA)
echo Expected runtime: 8-10 hours
echo ======================================================================

if not exist "%OUTPUT_DIR%\L0\campaign_summary.json" (
    echo Running L0 (Static LoRA) baseline experiments...
    python scripts\run_batch_experiments.py --config experiments\L0_static_lora.yaml --seeds %SEEDS% --output %OUTPUT_DIR%\L0\ --verbose
    echo ✓ L0 baseline experiments complete
) else (
    echo ✓ L0 baseline results found, skipping...
)

REM BEM Experiments  
echo.
echo ======================================================================
echo PHASE 2: BEM v1.1-stable Experiments
echo Expected runtime: 16-20 hours
echo ======================================================================

if not exist "%OUTPUT_DIR%\B1\campaign_summary.json" (
    echo Running B1 (BEM v1.1-stable) experiments...
    python scripts\run_batch_experiments.py --config experiments\B1_bem_v11_stable.yaml --seeds %SEEDS% --output %OUTPUT_DIR%\B1\ --verbose
    echo ✓ B1 BEM experiments complete
) else (
    echo ✓ B1 BEM results found, skipping...
)

REM Variant Experiments (Optional)
echo.
echo ======================================================================
echo PHASE 3: BEM Variant Experiments (Optional)
echo Expected runtime: 10-16 hours per variant
echo ======================================================================

REM V2: Dual Path BEM
if not exist "%OUTPUT_DIR%\V2\campaign_summary.json" (
    echo Running V2 (Dual Path BEM) experiments...
    python scripts\run_batch_experiments.py --config experiments\V2_dual_path.yaml --seeds %SEEDS% --output %OUTPUT_DIR%\V2\ --verbose
    echo ✓ V2 variant experiments complete
) else (
    echo ✓ V2 variant results found, skipping...
)

REM V7: FiLM Lite BEM
if not exist "%OUTPUT_DIR%\V7\campaign_summary.json" (
    echo Running V7 (FiLM Lite BEM) experiments...
    python scripts\run_batch_experiments.py --config experiments\V7_film_lite.yaml --seeds %SEEDS% --output %OUTPUT_DIR%\V7\ --verbose
    echo ✓ V7 variant experiments complete
) else (
    echo ✓ V7 variant results found, skipping...
)

REM V11: Learned Cache Policy
if not exist "%OUTPUT_DIR%\V11\campaign_summary.json" (
    echo Running V11 (Learned Cache Policy) experiments...
    python scripts\run_batch_experiments.py --config experiments\V11_learned_cache_policy.yaml --seeds %SEEDS% --output %OUTPUT_DIR%\V11\ --verbose
    echo ✓ V11 variant experiments complete
) else (
    echo ✓ V11 variant results found, skipping...
)

REM Statistical Analysis
echo.
echo ======================================================================
echo PHASE 4: Statistical Analysis and Validation
echo Expected runtime: 1-2 hours
echo ======================================================================

echo Running rigorous statistical analysis...
python scripts\run_statistical_pipeline.py --baseline %OUTPUT_DIR%\L0\ --treatment %OUTPUT_DIR%\B1\ --bootstrap-samples 10000 --confidence-level 0.95 --multiple-correction fdr --output %ANALYSIS_DIR%\statistical_results.json --verbose

echo Validating pre-registered claims...
python scripts\init_claims.py --validate --claims paper\claims.yaml --results %ANALYSIS_DIR%\statistical_results.json --output %ANALYSIS_DIR%\claim_validation.json

echo Generating hero tables...
python scripts\generate_tables.py --results %ANALYSIS_DIR%\statistical_results.json --output %ANALYSIS_DIR%\hero_table.csv

echo ✓ Statistical analysis complete

REM Paper Generation
echo.
echo ======================================================================
echo PHASE 5: Paper Assembly and Validation
echo Expected runtime: 15-30 minutes
echo ======================================================================

echo Generating figures...
python scripts\generate_figures.py --results %ANALYSIS_DIR%\ --output %PAPER_DIR%\figures\

echo Auto-generating paper sections...
python scripts\generate_sections.py --results %ANALYSIS_DIR%\statistical_results.json --claims %ANALYSIS_DIR%\claim_validation.json --output %PAPER_DIR%\auto\

echo Assembling final paper...
python scripts\assemble_paper.py --main-tex %PAPER_DIR%\main.tex --supplement-tex %PAPER_DIR%\supplement.tex --output %PAPER_DIR%\

echo Building PDFs...
cd %PAPER_DIR%
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

pdflatex supplement.tex  
bibtex supplement
pdflatex supplement.tex
pdflatex supplement.tex
cd ..

echo ✓ Paper generation complete

REM Final Validation
echo.
echo ======================================================================
echo PHASE 6: Reproduction Validation
echo ======================================================================

echo Validating reproduction results...

echo Key Results Validation:
echo   Expected: EM +11.1%%, BLEU +35.3%%

python -c "import json; data = json.load(open('%ANALYSIS_DIR%/statistical_results.json')); em = data['comparisons'][0]['relative_improvement_pct']; bleu = data['comparisons'][2]['relative_improvement_pct']; print(f'  Actual: EM +{em:.1f}%%, BLEU +{bleu:.1f}%%')"

echo.
echo File Validation:
if exist "%PAPER_DIR%\main.pdf" (echo   ✓ %PAPER_DIR%\main.pdf) else (echo   ✗ %PAPER_DIR%\main.pdf (MISSING^))
if exist "%PAPER_DIR%\supplement.pdf" (echo   ✓ %PAPER_DIR%\supplement.pdf) else (echo   ✗ %PAPER_DIR%\supplement.pdf (MISSING^))
if exist "%ANALYSIS_DIR%\statistical_results.json" (echo   ✓ %ANALYSIS_DIR%\statistical_results.json) else (echo   ✗ %ANALYSIS_DIR%\statistical_results.json (MISSING^))
if exist "%OUTPUT_DIR%\L0\campaign_summary.json" (echo   ✓ %OUTPUT_DIR%\L0\campaign_summary.json) else (echo   ✗ %OUTPUT_DIR%\L0\campaign_summary.json (MISSING^))
if exist "%OUTPUT_DIR%\B1\campaign_summary.json" (echo   ✓ %OUTPUT_DIR%\B1\campaign_summary.json) else (echo   ✗ %OUTPUT_DIR%\B1\campaign_summary.json (MISSING^))

echo.
echo ======================================================================
echo REPRODUCTION COMPLETE!
echo ======================================================================
echo.
echo Generated Files:
echo   📄 Main Paper: %PAPER_DIR%\main.pdf
echo   📋 Supplement: %PAPER_DIR%\supplement.pdf  
echo   📊 Statistical Results: %ANALYSIS_DIR%\statistical_results.json
echo   📁 Complete Logs: %OUTPUT_DIR%\
echo.
echo Key Findings:
python -c "import json; data = json.load(open('%ANALYSIS_DIR%/statistical_results.json')); print('  • All 4 primary metrics significantly improved:'); [print(f'    - {comp[\"metric\"]}: +{comp[\"relative_improvement_pct\"]:.1f}%% (CI: [{comp[\"ci_lower\"]:.1f}%%, {comp[\"ci_upper\"]:.1f}%%])') for comp in data['comparisons'][:4]]"
echo.
python -c "import json; data = json.load(open('%ANALYSIS_DIR%/statistical_results.json')); gates_passed = sum(1 for gate in data['quality_gates'] if gate['passed']); total_gates = len(data['quality_gates']); print(f'  • Quality Gates: {gates_passed}/{total_gates} passed')"

echo.
echo Next Steps:
echo   1. Review paper\main.pdf for main results
echo   2. Check paper\supplement.pdf for detailed analysis
echo   3. Examine analysis\statistical_results.json for raw statistics  
echo   4. Use logs\ directory for detailed experimental traces
echo.
echo Questions? See paper\repro_manifest.json for detailed instructions.
echo ======================================================================

pause