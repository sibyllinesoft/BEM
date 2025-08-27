"""
Simple test to verify Phase 3 imports work correctly.
This tests the module structure without requiring torch/transformers.
"""

import sys
import importlib.util

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, f"Module {module_name} not found"
        return True, f"Module {module_name} found"
    except Exception as e:
        return False, f"Import error for {module_name}: {e}"

def main():
    """Test Phase 3 module structure."""
    print("🔍 Testing Phase 3 Module Structure")
    print("=" * 40)
    
    # Test core modules exist
    modules_to_test = [
        'bem',
        'bem.retrieval',
        'bem.retrieval_features', 
        'bem.retrieval_bem',
        'bem.retrieval_training',
        'bem.controller',
        'bem.hierarchical_bem'
    ]
    
    results = []
    for module_name in modules_to_test:
        success, message = test_import(module_name)
        results.append((success, message))
        status = "✅" if success else "❌"
        print(f"{status} {message}")
    
    print("\n" + "=" * 40)
    
    # Summary
    success_count = sum(1 for success, _ in results if success)
    total_count = len(results)
    
    print(f"📊 Import Test Results: {success_count}/{total_count} modules found")
    
    if success_count == total_count:
        print("🎉 All Phase 3 modules are properly structured!")
        print("\nKey Phase 3 components available:")
        print("  • bem.retrieval - Micro-retriever with FAISS")
        print("  • bem.retrieval_features - Coverage/consistency features")
        print("  • bem.retrieval_bem - Retrieval-aware BEM integration") 
        print("  • bem.retrieval_training - Training with retrieval losses")
        return True
    else:
        print("⚠️  Some modules missing - check file structure")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)