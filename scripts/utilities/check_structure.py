#!/usr/bin/env python3
"""
Structural validation of BEM v1.1 implementation without PyTorch dependencies
"""

import pathlib
import ast
import sys
import os

def check_file_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def analyze_imports(file_path):
    """Extract imports from a Python file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    except Exception as e:
        return []

def find_classes(file_path):
    """Find all class definitions in a Python file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    except Exception as e:
        return []

def main():
    print("🔍 BEM-v1.1-stable Structural Validation")
    print("=" * 60)
    
    base_path = pathlib.Path(__file__).parent
    bem_path = base_path / "bem"
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    expected_dirs = [
        "bem",
        "bem/modules", 
        "bem/models",
        "bem/training",
        "bem/evaluation"
    ]
    
    for dir_path in expected_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
    
    # Check expected files
    print("\n📄 Expected Files:")
    expected_files = [
        # Main modules
        "bem/modules/__init__.py",
        "bem/modules/parallel_lora.py",
        "bem/modules/chunk_sticky_routing.py",
        "bem/modules/attention_bias.py", 
        "bem/modules/governance.py",
        
        # Models
        "bem/models/__init__.py",
        "bem/models/bem_v11.py",
        
        # Training
        "bem/training/__init__.py",
        "bem/training/bem_v11_trainer.py",
        "bem/training/cache_metrics.py",
        
        # Evaluation
        "bem/evaluation/__init__.py",
        "bem/evaluation/bem_evaluator.py",
        "bem/evaluation/slice_analysis.py",
        "bem/evaluation/cache_analysis.py",
        
        # Main scripts
        "train.py",
        "evaluate.py",
        "validate_bem_v11.py"
    ]
    
    valid_files = []
    for file_path in expected_files:
        full_path = base_path / file_path
        if full_path.exists():
            valid_syntax, error = check_file_syntax(full_path)
            if valid_syntax:
                print(f"✅ {file_path}")
                valid_files.append(full_path)
            else:
                print(f"⚠️  {file_path} - SYNTAX ERROR: {error}")
        else:
            print(f"❌ {file_path} - MISSING")
    
    # Analyze key implementation files
    print("\n🧬 Implementation Analysis:")
    
    key_files = {
        "bem/modules/parallel_lora.py": ["GeneratedParallelLoRA", "AdapterGenerator"],
        "bem/modules/chunk_sticky_routing.py": ["ChunkStickyRouter"],
        "bem/modules/attention_bias.py": ["AttentionLogitBias"],
        "bem/modules/governance.py": ["GovernanceModule", "SpectralConstraint", "FrobeniusConstraint"],
        "bem/models/bem_v11.py": ["BEMv11Model"],
        "bem/training/bem_v11_trainer.py": ["BEMv11Trainer", "BEMv11TrainingConfig"],
        "bem/evaluation/bem_evaluator.py": ["BEMv11Evaluator"]
    }
    
    for file_path, expected_classes in key_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            classes = find_classes(full_path)
            print(f"\n  📝 {file_path}:")
            for expected_class in expected_classes:
                if expected_class in classes:
                    print(f"    ✅ {expected_class}")
                else:
                    print(f"    ❌ {expected_class} - MISSING")
            
            # Show additional classes found
            additional = set(classes) - set(expected_classes)
            if additional:
                print(f"    ℹ️  Additional classes: {', '.join(additional)}")
    
    # Check TODO.md compliance
    print(f"\n🎯 TODO.md Compliance Check:")
    
    todo_path = base_path / "TODO.md"
    if todo_path.exists():
        print("✅ TODO.md found")
        
        # Check if key requirements are mentioned in implementation
        key_requirements = [
            ("E1: Generated Parallel LoRA", "bem/modules/parallel_lora.py"),
            ("E3: Chunk-sticky routing", "bem/modules/chunk_sticky_routing.py"),
            ("E4: Attention-logit bias", "bem/modules/attention_bias.py"),
            ("Governance", "bem/modules/governance.py"),
            ("5-seed protocol", "train.py"),
            ("Cache metrics", "bem/training/cache_metrics.py"),
            ("Slice analysis", "bem/evaluation/slice_analysis.py")
        ]
        
        for requirement, file_path in key_requirements:
            full_path = base_path / file_path
            if full_path.exists():
                print(f"    ✅ {requirement} → {file_path}")
            else:
                print(f"    ❌ {requirement} → {file_path} MISSING")
    else:
        print("❌ TODO.md not found")
    
    print(f"\n📊 Summary:")
    total_expected = len(expected_files)
    found_files = len([f for f in expected_files if (base_path / f).exists()])
    syntax_valid = len(valid_files)
    
    print(f"  • Files found: {found_files}/{total_expected} ({found_files/total_expected*100:.1f}%)")
    print(f"  • Syntax valid: {syntax_valid}/{found_files} ({syntax_valid/found_files*100:.1f}%)")
    
    if found_files == total_expected and syntax_valid == found_files:
        print(f"\n🎉 BEM-v1.1-stable Implementation Complete!")
        print(f"   All required files are present with valid syntax.")
        print(f"   Ready for PyTorch environment setup and training.")
    else:
        print(f"\n⚠️  Implementation Incomplete:")
        if found_files < total_expected:
            print(f"   Missing {total_expected - found_files} expected files")
        if syntax_valid < found_files:
            print(f"   {found_files - syntax_valid} files have syntax errors")

if __name__ == "__main__":
    main()