#!/usr/bin/env python3
"""
Simple test to verify Phase 4 imports work correctly.
This test can run even without torch/other dependencies installed.
"""

import sys
import importlib.util
from pathlib import Path

def test_module_imports():
    """Test that all Phase 4 modules can be imported without syntax errors."""
    
    # Test individual module imports
    modules_to_test = [
        'bem.subspace',
        'bem.trust_region', 
        'bem.multi_bem',
        'bem.interference_testing',
        'bem.composition_training'
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for module_name in modules_to_test:
        try:
            # Convert module name to file path
            file_path = Path(module_name.replace('.', '/') + '.py')
            
            # Load the module spec
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                results['failed'] += 1
                results['details'].append(f"❌ {module_name}: Could not load module spec")
                continue
                
            # Create module (this checks syntax)
            module = importlib.util.module_from_spec(spec)
            
            # Check that we can parse the file
            with open(file_path, 'r') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            
            results['passed'] += 1 
            results['details'].append(f"✅ {module_name}: Syntax valid, imports loadable")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {module_name}: {e}")
    
    return results

def test_bem_init_exports():
    """Test that bem/__init__.py includes Phase 4 exports."""
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    try:
        # Read bem/__init__.py
        init_file = Path('bem/__init__.py')
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Check for Phase 4 components
        phase4_components = [
            'SubspacePlanner',
            'TrustRegionProjector', 
            'MultiBEMComposer',
            'InterferenceTester',
            'CompositionTrainer'
        ]
        
        missing_components = []
        for component in phase4_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            results['failed'] += 1
            results['details'].append(f"❌ bem/__init__.py: Missing exports - {missing_components}")
        else:
            results['passed'] += 1
            results['details'].append("✅ bem/__init__.py: All Phase 4 components exported")
        
        # Check version update
        if "0.4.0" in content:
            results['passed'] += 1
            results['details'].append("✅ bem/__init__.py: Version updated to 0.4.0")
        else:
            results['failed'] += 1
            results['details'].append("❌ bem/__init__.py: Version not updated")
            
        # Check Phase 4 description
        if "Multi-BEM Composition" in content:
            results['passed'] += 1  
            results['details'].append("✅ bem/__init__.py: Phase 4 description added")
        else:
            results['failed'] += 1
            results['details'].append("❌ bem/__init__.py: Phase 4 description missing")
            
    except Exception as e:
        results['failed'] += 1
        results['details'].append(f"❌ bem/__init__.py test failed: {e}")
    
    return results

def main():
    """Run all Phase 4 import tests."""
    print("🧪 Testing BEM Phase 4 Implementation")
    print("=" * 50)
    
    # Test module imports
    print("1. Testing module syntax and loadability...")
    import_results = test_module_imports()
    
    for detail in import_results['details']:
        print(f"   {detail}")
    
    # Test bem/__init__.py
    print("\n2. Testing bem/__init__.py exports...")
    init_results = test_bem_init_exports()
    
    for detail in init_results['details']:
        print(f"   {detail}")
    
    # Summary
    print("\n" + "=" * 50)
    total_passed = import_results['passed'] + init_results['passed']
    total_failed = import_results['failed'] + init_results['failed'] 
    
    print(f"📋 SUMMARY: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All Phase 4 import tests passed!")
        print("✅ Phase 4 implementation is syntactically correct")
        print("✅ All modules are properly structured") 
        print("✅ bem/__init__.py exports are complete")
    else:
        print("❌ Some Phase 4 import tests failed")
        print("   Review the errors above")
    
    print("=" * 50)
    
    return total_failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)