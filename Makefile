# BEM Repository Development Makefile
# 
# This Makefile provides common development tasks for the BEM repository.
# Use 'make help' to see all available commands.

.PHONY: help install install-dev clean test test-cov test-fast test-slow test-integration test-gpu \
        lint format type-check security security-full docs docs-serve build publish validate \
        pre-commit docker-build docker-test profile benchmark \
        setup-dev setup-models env-check ci-local release-validate release-ready final-check \
        build-release docs-deploy social-assets

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m
BOLD := \033[1m

# Default target
help:
	@echo "$(BOLD)$(BLUE)BEM Repository Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Setup Commands:$(RESET)"
	@echo "  setup-dev     Complete development environment setup"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  setup-models  Download and setup model artifacts"
	@echo "  env-check     Check development environment status"
	@echo ""
	@echo "$(BOLD)Development Commands:$(RESET)"
	@echo "  clean         Remove build artifacts and cache files"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run linting checks (flake8)"
	@echo "  type-check    Run type checking with mypy"
	@echo "  security      Run basic security checks"
	@echo "  security-full Run comprehensive security analysis"
	@echo "  validate      Run full validation suite (format + lint + type + security + test)"
	@echo ""
	@echo "$(BOLD)Testing Commands:$(RESET)"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run fast tests only"
	@echo "  test-slow     Run slow/integration tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  test-integration Run integration tests (requires services)"
	@echo "  test-gpu      Run GPU tests (requires CUDA)"
	@echo "  benchmark     Run performance benchmarks"
	@echo ""
	@echo "$(BOLD)Documentation Commands:$(RESET)"
	@echo "  docs          Build documentation"
	@echo "  docs-serve    Serve documentation locally"
	@echo "  docs-check    Check documentation quality"
	@echo ""
	@echo "$(BOLD)Docker Commands:$(RESET)"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-test   Run tests in Docker containers"
	@echo "  docker-up     Start development services"
	@echo "  docker-down   Stop development services"
	@echo ""
	@echo "$(BOLD)Release Commands:$(RESET)"
	@echo "  build         Build distribution packages"
	@echo "  build-release Complete release build (packages + Docker + docs)"
	@echo "  publish       Publish to PyPI (requires authentication)"
	@echo "  version       Show current version information"
	@echo "  release-validate  Run release validation script"
	@echo "  final-check   Final comprehensive validation before release"
	@echo "  release-ready Complete release readiness validation"
	@echo ""
	@echo "$(BOLD)Utility Commands:$(RESET)"
	@echo "  pre-commit    Setup pre-commit hooks"
	@echo "  profile       Run performance profiling"
	@echo "  ci-local      Simulate CI pipeline locally"
	@echo "  dev           Quick development cycle (format + lint + test-fast)"

# Python and pip commands
PYTHON := python3
PIP := pip3

# Check if we're in a virtual environment
VENV_PATH := .venv
ifeq ($(shell test -d $(VENV_PATH) && echo 1), 1)
    PYTHON := $(VENV_PATH)/bin/python
    PIP := $(VENV_PATH)/bin/pip
endif

# ============================================================================
# Setup Commands
# ============================================================================

setup-dev:
	@echo "$(BLUE)Setting up complete development environment...$(RESET)"
	$(PYTHON) scripts/dev/setup-dev-environment.py $(ARGS)
	@echo "$(GREEN)✓ Development environment setup complete!$(RESET)"

install:
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(RESET)"

install-dev:
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ Development dependencies installed$(RESET)"

setup-models:
	@echo "$(BLUE)Downloading model artifacts...$(RESET)"
	$(PYTHON) scripts/setup/download_models.py --setup-all
	@echo "$(GREEN)✓ Model artifacts downloaded$(RESET)"

env-check:
	@echo "$(BOLD)Development Environment Status$(RESET)"
	@echo "================================"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Virtual env: $(if $(shell test -d $(VENV_PATH) && echo 1),$(GREEN)Active$(RESET),$(YELLOW)Not active$(RESET))"
	@echo ""
	@echo "$(BOLD)Tool Availability:$(RESET)"
	@which black >/dev/null 2>&1 && echo "  black: $(GREEN)✓$(RESET)" || echo "  black: $(RED)✗$(RESET)"
	@which flake8 >/dev/null 2>&1 && echo "  flake8: $(GREEN)✓$(RESET)" || echo "  flake8: $(RED)✗$(RESET)"
	@which mypy >/dev/null 2>&1 && echo "  mypy: $(GREEN)✓$(RESET)" || echo "  mypy: $(RED)✗$(RESET)"
	@which pytest >/dev/null 2>&1 && echo "  pytest: $(GREEN)✓$(RESET)" || echo "  pytest: $(RED)✗$(RESET)"
	@which pre-commit >/dev/null 2>&1 && echo "  pre-commit: $(GREEN)✓$(RESET)" || echo "  pre-commit: $(RED)✗$(RESET)"

# ============================================================================
# Development Commands
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning build artifacts and cache files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage*" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.log" -path "./logs/*" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup completed$(RESET)"

format:
	@echo "$(BLUE)Formatting code with black and isort...$(RESET)"
	black src/ tests/ scripts/ --line-length=88 --target-version py39
	isort src/ tests/ scripts/ --profile black --line-length 88
	@echo "$(GREEN)✓ Code formatting completed$(RESET)"

lint:
	@echo "$(BLUE)Running linting checks...$(RESET)"
	flake8 src/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503 --statistics
	@echo "$(GREEN)✓ Linting checks completed$(RESET)"

type-check:
	@echo "$(BLUE)Running type checking with mypy...$(RESET)"
	mypy src/ --config-file mypy.ini --install-types --non-interactive
	@echo "$(GREEN)✓ Type checking completed$(RESET)"

security:
	@echo "$(BLUE)Running basic security checks...$(RESET)"
	bandit -r src/ -f json -o security-report.json || true
	bandit -r src/ --severity-level medium
	safety check --json --output safety-report.json || true
	safety check --short-report
	@echo "$(YELLOW)Security reports: security-report.json, safety-report.json$(RESET)"

security-full:
	@echo "$(BLUE)Running comprehensive security analysis...$(RESET)"
	bandit -r src/ tests/ scripts/ -f json -o bandit-full-report.json
	bandit -r src/ tests/ scripts/ --severity-level low
	safety check --full-report --output safety-full-report.txt
	semgrep --config=auto src/ --json --output=semgrep-report.json || true
	@echo "$(YELLOW)Full security reports generated$(RESET)"

validate: clean format lint type-check security test-cov
	@echo "$(GREEN)$(BOLD)✓ All validation checks completed successfully!$(RESET)"

# ============================================================================
# Testing Commands  
# ============================================================================

test:
	@echo "$(BLUE)Running all tests...$(RESET)"
	pytest tests/ -v --tb=short --maxfail=5

test-fast:
	@echo "$(BLUE)Running fast tests only...$(RESET)"
	pytest tests/ -v -m "not slow and not integration and not gpu" --tb=line

test-slow:
	@echo "$(BLUE)Running slow and integration tests...$(RESET)"
	pytest tests/ -v -m "slow or integration" --tb=short --timeout=600

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing --cov-fail-under=80

test-integration:
	@echo "$(BLUE)Running integration tests...$(RESET)"
	@echo "$(YELLOW)Starting test services...$(RESET)"
	docker-compose -f docker-compose.test.yml up -d postgres-test redis-test
	sleep 10  # Wait for services to start
	pytest tests/ -v -m integration --tb=short
	docker-compose -f docker-compose.test.yml down

test-gpu:
	@echo "$(BLUE)Running GPU tests...$(RESET)"
	@echo "$(YELLOW)Note: Requires CUDA-capable GPU$(RESET)"
	pytest tests/ -v -m gpu --tb=short --timeout=900

benchmark:
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	pytest tests/ -v -m "not slow" --benchmark-only --benchmark-json=benchmark-results.json
	@echo "$(YELLOW)Benchmark results: benchmark-results.json$(RESET)"

# ============================================================================
# Documentation Commands
# ============================================================================

docs:
	@echo "$(BLUE)Building documentation...$(RESET)"
	@if [ ! -f mkdocs.yml ]; then \
		echo "$(YELLOW)Creating basic mkdocs.yml...$(RESET)"; \
		$(PYTHON) -c "from scripts.dev.setup_dev_environment import create_basic_mkdocs_config; create_basic_mkdocs_config()"; \
	fi
	mkdocs build --strict --verbose

docs-serve:
	@echo "$(BLUE)Serving documentation locally...$(RESET)"
	mkdocs serve --dev-addr 127.0.0.1:8000

docs-check:
	@echo "$(BLUE)Checking documentation quality...$(RESET)"
	$(PYTHON) -c "
import ast
import pathlib

def count_docstrings():
    total, documented = 0, 0
    for py_file in pathlib.Path('src').rglob('*.py'):
        if py_file.name.startswith('_') and py_file.name != '__init__.py':
            continue
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith('_'):
                        total += 1
                        if ast.get_docstring(node):
                            documented += 1
        except:
            continue
    if total > 0:
        coverage = (documented / total) * 100
        print(f'Documentation coverage: {coverage:.1f}% ({documented}/{total})')
    else:
        print('No functions found for documentation coverage')

count_docstrings()
	"

# ============================================================================
# Docker Commands
# ============================================================================

docker-build:
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build -t bem:latest .
	docker build -f Dockerfile.test -t bem:test .

docker-test:
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down

docker-up:
	@echo "$(BLUE)Starting development services...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Services started. Check with 'docker-compose ps'$(RESET)"

docker-down:
	@echo "$(BLUE)Stopping development services...$(RESET)"
	docker-compose down
	@echo "$(GREEN)Services stopped$(RESET)"

# ============================================================================
# Release Commands
# ============================================================================

version:
	@echo "$(BOLD)Version Information$(RESET)"
	@$(PYTHON) -c "
import toml
config = toml.load('pyproject.toml')
print(f'Package: {config[\"project\"][\"name\"]}')
print(f'Version: {config[\"project\"][\"version\"]}')
print(f'Description: {config[\"project\"][\"description\"]}')
"

build: clean
	@echo "$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build --sdist --wheel
	@echo "$(GREEN)✓ Build completed. Files in dist/$(RESET)"
	@ls -la dist/

publish: build
	@echo "$(BLUE)Publishing to PyPI...$(RESET)"
	@echo "$(YELLOW)This will publish to PyPI. Continue? [y/N]$(RESET)"
	@read confirm && [ "$$confirm" = "y" ] || exit 1
	$(PYTHON) -m twine upload dist/*

# ============================================================================
# Release Validation Commands
# ============================================================================

release-validate:
	@echo "$(BLUE)Running release validation...$(RESET)"
	$(PYTHON) scripts/validate_release.py --comprehensive
	@echo "$(GREEN)✓ Release validation completed$(RESET)"

final-check:
	@echo "$(BLUE)Running final comprehensive validation...$(RESET)"
	$(PYTHON) scripts/final_release_validation.py
	@echo "$(GREEN)✓ Final validation completed$(RESET)"

release-ready: clean validate security-full release-validate final-check
	@echo "$(GREEN)$(BOLD)🚀 REPOSITORY IS RELEASE-READY!$(RESET)"
	@echo "$(GREEN)All validation checks passed successfully.$(RESET)"
	@echo "$(BLUE)Next steps:$(RESET)"
	@echo "  1. Create release branch: git checkout -b release/vX.Y.Z"
	@echo "  2. Update CHANGELOG.md with release notes"
	@echo "  3. Run: make build-release"
	@echo "  4. Test installation: pip install dist/*.whl"
	@echo "  5. Create GitHub release with assets"

build-release: build docker-build docs-deploy
	@echo "$(BLUE)Building complete release assets...$(RESET)"
	@echo "$(BOLD)Package artifacts:$(RESET)"
	@ls -la dist/
	@echo ""
	@echo "$(BOLD)Docker images:$(RESET)"
	@docker images bem:latest
	@echo ""
	@echo "$(GREEN)✓ Complete release build finished!$(RESET)"
	@echo "$(BLUE)Upload to GitHub Releases:$(RESET)"
	@echo "  - dist/*.whl (Python wheel)"
	@echo "  - dist/*.tar.gz (Source distribution)"
	@echo "  - Docker: docker push your-registry/bem:latest"

docs-deploy:
	@echo "$(BLUE)Deploying documentation...$(RESET)"
	@if [ -d "docs/_build/html" ]; then \
		echo "$(GREEN)✓ Documentation built and ready for deployment$(RESET)"; \
	else \
		echo "$(YELLOW)Building documentation first...$(RESET)"; \
		$(MAKE) docs; \
	fi

social-assets:
	@echo "$(BLUE)Generating social media assets...$(RESET)"
	@echo "$(YELLOW)Social media assets available in marketing/$(RESET)"
	@echo "  - Repository badges: marketing/badges_and_analytics.md"
	@echo "  - Marketing guide: marketing/README.md"
	@echo "  - Release template: .github/RELEASE_TEMPLATE.md"

# ============================================================================
# Utility Commands
# ============================================================================

pre-commit:
	@echo "$(BLUE)Setting up pre-commit hooks...$(RESET)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit install --hook-type pre-push
	@echo "$(GREEN)✓ Pre-commit hooks installed$(RESET)"

profile:
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats scripts/demos/demo_simple_bem.py
	$(PYTHON) -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
print('\nTop functions by total time:')
stats.sort_stats('tottime')
stats.print_stats(10)
"

ci-local:
	@echo "$(BLUE)Simulating CI pipeline locally...$(RESET)"
	@echo "$(BOLD)Step 1: Code formatting$(RESET)"
	@$(MAKE) format
	@echo "$(BOLD)Step 2: Linting$(RESET)"
	@$(MAKE) lint
	@echo "$(BOLD)Step 3: Type checking$(RESET)"
	@$(MAKE) type-check
	@echo "$(BOLD)Step 4: Security checks$(RESET)"
	@$(MAKE) security
	@echo "$(BOLD)Step 5: Fast tests$(RESET)"
	@$(MAKE) test-fast
	@echo "$(BOLD)Step 6: Coverage tests$(RESET)"
	@$(MAKE) test-cov
	@echo "$(GREEN)$(BOLD)✓ Local CI pipeline completed successfully!$(RESET)"

dev: format lint test-fast
	@echo "$(GREEN)$(BOLD)✓ Development cycle completed!$(RESET)"

# ============================================================================
# Research and Experimentation
# ============================================================================

research: setup-models
	@echo "$(BLUE)Starting Jupyter Lab for research...$(RESET)"
	jupyter lab --notebook-dir=experiments/ --ip=0.0.0.0 --port=8888

experiment:
	@echo "$(BLUE)Running experiment: $(EXP)$(RESET)"
	@if [ -z "$(EXP)" ]; then \
		echo "$(RED)Error: Please specify experiment with EXP=experiment_name$(RESET)"; \
		echo "Available experiments:"; \
		ls experiments/*.yml experiments/*.yaml 2>/dev/null | sed 's/.*\///g' | sed 's/\.(yml\|yaml)$$//g' | sort; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_experiment.py --config experiments/$(EXP).yml

# ============================================================================
# Model and Data Management
# ============================================================================

list-models:
	@echo "$(BOLD)Available Models$(RESET)"
	$(PYTHON) scripts/setup/download_models.py --list

update-models:
	@echo "$(BLUE)Updating all model artifacts...$(RESET)"
	$(PYTHON) scripts/setup/download_models.py --setup-all --force

clean-models:
	@echo "$(YELLOW)Cleaning model cache...$(RESET)"
	@echo "This will remove all cached models. Continue? [y/N]"
	@read confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf models/*/model_cache/
	rm -rf models/*/tokenizer_cache/
	@echo "$(GREEN)✓ Model cache cleaned$(RESET)"

# ============================================================================
# Project Statistics and Information
# ============================================================================

stats:
	@echo "$(BOLD)$(BLUE)BEM Project Statistics$(RESET)"
	@echo "======================="
	@echo "$(BOLD)Code Statistics:$(RESET)"
	@echo -n "  Python files: "; find src/ -name "*.py" | wc -l
	@echo -n "  Lines of code: "; find src/ -name "*.py" | xargs wc -l | tail -1 | awk '{print $$1}'
	@echo -n "  Test files: "; find tests/ -name "*.py" | wc -l  
	@echo -n "  Test lines: "; find tests/ -name "*.py" | xargs wc -l | tail -1 | awk '{print $$1}' 2>/dev/null || echo "0"
	@echo ""
	@echo "$(BOLD)Configuration Files:$(RESET)"
	@echo -n "  Experiments: "; find experiments/ -name "*.yml" -o -name "*.yaml" | wc -l 2>/dev/null || echo "0"
	@echo -n "  Documentation: "; find docs/ -name "*.md" -o -name "*.rst" | wc -l 2>/dev/null || echo "0"
	@echo ""
	@echo "$(BOLD)Git Statistics:$(RESET)"
	@git log --oneline | wc -l | awk '{print "  Commits: " $$1}' 2>/dev/null || echo "  Commits: N/A"
	@git branch -r | wc -l | awk '{print "  Remote branches: " $$1}' 2>/dev/null || echo "  Remote branches: N/A"

info:
	@echo "$(BOLD)$(BLUE)BEM Development Environment Info$(RESET)"
	@echo "======================================="
	@echo "$(BOLD)Project:$(RESET)"
	@$(PYTHON) -c "
import toml
try:
    config = toml.load('pyproject.toml')['project']
    print(f'  Name: {config[\"name\"]}')
    print(f'  Version: {config[\"version\"]}')
    print(f'  Description: {config[\"description\"]}')
    print(f'  Python: {config[\"requires-python\"]}')
except:
    print('  Could not read project info')
"
	@echo ""
	@echo "$(BOLD)Environment:$(RESET)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pip: $(shell $(PIP) --version)"
	@echo "  Working directory: $(shell pwd)"
	@echo "  Virtual env: $(if $(shell test -d $(VENV_PATH) && echo 1),Active ($(VENV_PATH)),Not active)"

# Help for specific categories
help-dev:
	@echo "$(BOLD)$(BLUE)Development Workflow$(RESET)"
	@echo "===================="
	@echo "1. $(BOLD)Setup:$(RESET) make setup-dev"
	@echo "2. $(BOLD)Daily dev:$(RESET) make dev (format + lint + test-fast)"  
	@echo "3. $(BOLD)Before commit:$(RESET) make validate"
	@echo "4. $(BOLD)Before push:$(RESET) make ci-local"
	@echo ""
	@echo "$(BOLD)Quick Commands:$(RESET)"
	@echo "  make dev           # Fast development cycle"
	@echo "  make test-fast     # Run only fast tests" 
	@echo "  make format lint   # Format and lint code"
	@echo "  make clean         # Clean build artifacts"

help-test:
	@echo "$(BOLD)$(BLUE)Testing Options$(RESET)"
	@echo "==============="
	@echo "  make test          # All tests"
	@echo "  make test-fast     # Fast tests only"
	@echo "  make test-slow     # Integration/slow tests"
	@echo "  make test-cov      # With coverage report"
	@echo "  make test-gpu      # GPU-specific tests"
	@echo "  make benchmark     # Performance benchmarks"
	@echo "  make docker-test   # Tests in Docker"

# Default target aliases
all: validate
check: validate
fast: dev
full: ci-local