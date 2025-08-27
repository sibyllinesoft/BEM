#!/bin/bash
# BEM Release Preparation - Main Entry Point
# This script orchestrates the complete release preparation process

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Python
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    # Check Git
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    # Check Git LFS
    if ! command_exists git-lfs; then
        print_warning "Git LFS not found - will skip LFS setup"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_error "Please install them and try again."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l 2>/dev/null || echo "0") != "1" ]]; then
        print_error "Python 3.9+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed"
    else
        print_warning "requirements.txt not found"
    fi
    
    # Install dev requirements if they exist
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Dev requirements installed"
    fi
}

# Function to show help
show_help() {
    cat << EOF
BEM Release Preparation Script

Usage: $0 [OPTIONS]

OPTIONS:
    --dry-run           Run in dry-run mode without making changes
    --skip-backup      Skip creating backup of current state
    --skip-tests       Skip running test suite
    --help, -h         Show this help message

PHASES:
    1. Repository Cleanup    - Remove temporary and cache files
    2. Documentation Setup   - Create and organize documentation
    3. Git LFS Setup        - Configure Git LFS for large files
    4. Test Suite           - Run comprehensive tests
    5. Security Scan        - Run security and safety checks
    6. Performance Check    - Validate performance characteristics
    7. Package Build        - Build Python package
    8. Final Validation     - Final checks before release

EXAMPLES:
    # Full release preparation
    $0
    
    # Dry run to see what would be done
    $0 --dry-run
    
    # Skip backup and tests for quick check
    $0 --skip-backup --skip-tests

EOF
}

# Parse command line arguments
DRY_RUN=""
SKIP_BACKUP=""
SKIP_TESTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP="--skip-backup"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="--skip-tests"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting BEM Release Preparation"
    echo "=================================="
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup virtual environment
    setup_venv
    
    # Make sure scripts are executable
    chmod +x scripts/release_preparation/*.sh 2>/dev/null || true
    
    # Run the Python master script
    PYTHON_ARGS="$DRY_RUN $SKIP_BACKUP"
    
    print_status "Running Python master script..."
    python scripts/release_preparation/release_master_script.py \
        --project-root "$PROJECT_ROOT" \
        --report-file "release_preparation_report.json" \
        $PYTHON_ARGS
    
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        print_success "Release preparation completed successfully!"
        
        echo ""
        echo "=================================="
        echo "🚀 BEM Release Preparation Complete"
        echo "=================================="
        echo ""
        echo "📋 Next Steps:"
        echo "1. Review the release preparation report: release_preparation_report.json"
        echo "2. Review generated/updated files"
        echo "3. Commit any remaining changes:"
        echo "   git add ."
        echo "   git commit -m 'Prepare for release'"
        echo "4. Create and push release tag:"
        echo "   git tag -a v2.0.0 -m 'Release v2.0.0'"
        echo "   git push origin main --tags"
        echo "5. Create GitHub release from the pushed tag"
        echo "6. Publish package (if desired):"
        echo "   twine upload dist/*"
        echo ""
        echo "🎉 Your BEM repository is now ready for release!"
        
    else
        print_error "Release preparation failed!"
        echo ""
        echo "Please check the output above for error details."
        echo "You may need to fix issues and run the script again."
        exit 1
    fi
}

# Run main function
main "$@"