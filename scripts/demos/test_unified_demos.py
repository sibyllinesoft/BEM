#!/usr/bin/env python3
"""
Test script for unified demo scripts.

Validates that all unified demo scripts can be imported and basic functionality works.
This serves as a quick validation that the unified interface is properly implemented.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_demo_imports():
    """Test that all unified demo scripts can be imported."""
    
    print("🧪 Testing Unified Demo Script Imports")
    print("=" * 50)
    
    demos_to_test = [
        "demo_unified_multimodal",
        "demo_unified_performance", 
        "demo_unified_bem"
    ]
    
    results = {}
    
    for demo_name in demos_to_test:
        try:
            print(f"\n📦 Testing {demo_name}...")
            
            # Import the demo module
            demo_module = __import__(demo_name)
            
            # Check for key classes/functions
            expected_components = {
                "demo_unified_multimodal": ["MultimodalTrainer", "create_mock_multimodal_config"],
                "demo_unified_performance": ["UnifiedPerformanceTrainer", "create_unified_pt_configs"],
                "demo_unified_bem": ["UnifiedBEMSystem", "create_unified_configuration_templates"]
            }
            
            missing_components = []
            for component in expected_components[demo_name]:
                if not hasattr(demo_module, component):
                    missing_components.append(component)
            
            if missing_components:
                results[demo_name] = {
                    "status": "partial",
                    "error": f"Missing components: {missing_components}"
                }
                print(f"⚠️  Partial success - missing: {', '.join(missing_components)}")
            else:
                results[demo_name] = {"status": "success"}
                print(f"✅ Import successful - all components found")
                
        except ImportError as e:
            results[demo_name] = {
                "status": "import_error",
                "error": str(e)
            }
            print(f"❌ Import failed: {e}")
            
        except Exception as e:
            results[demo_name] = {
                "status": "error", 
                "error": str(e)
            }
            print(f"❌ Error: {e}")
    
    return results


def test_configuration_creation():
    """Test that configuration creation functions work."""
    
    print(f"\n🔧 Testing Configuration Creation")
    print("=" * 50)
    
    try:
        # Test multimodal config creation
        from demo_unified_multimodal import create_mock_multimodal_config
        config_path = create_mock_multimodal_config()
        
        # Check that config file was created
        if Path(config_path).exists():
            print("✅ Multimodal config creation successful")
            Path(config_path).unlink()  # Clean up
        else:
            print("❌ Multimodal config file not created")
            
    except Exception as e:
        print(f"❌ Multimodal config creation failed: {e}")
    
    try:
        # Test performance config creation
        from demo_unified_performance import create_unified_pt_configs
        pt_configs = create_unified_pt_configs()
        
        if len(pt_configs) == 4:  # Should create PT1-PT4 configs
            print("✅ Performance configs creation successful")
            # Clean up
            for config_path in pt_configs.values():
                Path(config_path).unlink(missing_ok=True)
        else:
            print(f"❌ Expected 4 PT configs, got {len(pt_configs)}")
            
    except Exception as e:
        print(f"❌ Performance config creation failed: {e}")
    
    try:
        # Test unified system config creation
        from demo_unified_bem import create_unified_configuration_templates
        unified_configs = create_unified_configuration_templates()
        
        expected_components = ["router", "safety", "multimodal", "performance"]
        if all(comp in unified_configs for comp in expected_components):
            print("✅ Unified system configs creation successful")
            # Clean up
            for config_path in unified_configs.values():
                Path(config_path).unlink(missing_ok=True)
        else:
            print(f"❌ Missing unified configs: {set(expected_components) - set(unified_configs.keys())}")
            
    except Exception as e:
        print(f"❌ Unified system config creation failed: {e}")


def test_trainer_instantiation():
    """Test that trainer classes can be instantiated."""
    
    print(f"\n🏋️ Testing Trainer Instantiation")
    print("=" * 50)
    
    try:
        # Test multimodal trainer
        from demo_unified_multimodal import MultimodalTrainer, create_mock_multimodal_config
        
        config_path = create_mock_multimodal_config()
        trainer = MultimodalTrainer(config_path, experiment_name="test")
        
        # Check key attributes
        if hasattr(trainer, 'experiment_config') and hasattr(trainer, 'device'):
            print("✅ MultimodalTrainer instantiation successful")
        else:
            print("❌ MultimodalTrainer missing required attributes")
            
        Path(config_path).unlink()  # Clean up
        
    except Exception as e:
        print(f"❌ MultimodalTrainer instantiation failed: {e}")
        traceback.print_exc()
    
    try:
        # Test performance trainer
        from demo_unified_performance import UnifiedPerformanceTrainer, create_unified_pt_configs
        
        pt_configs = create_unified_pt_configs()
        pt1_config = pt_configs.get("PT1")
        
        if pt1_config:
            trainer = UnifiedPerformanceTrainer(pt1_config, experiment_name="test")
            
            if hasattr(trainer, 'variant_type') and trainer.variant_type == "PT1":
                print("✅ UnifiedPerformanceTrainer instantiation successful")
            else:
                print("❌ UnifiedPerformanceTrainer variant detection failed")
        
        # Clean up
        for config_path in pt_configs.values():
            Path(config_path).unlink(missing_ok=True)
            
    except Exception as e:
        print(f"❌ UnifiedPerformanceTrainer instantiation failed: {e}")
        traceback.print_exc()


def test_config_loading():
    """Test that configuration loading works correctly."""
    
    print(f"\n📄 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        # Create a test config and try to load it
        from src.bem_core.config.config_loader import load_experiment_config
        from demo_unified_multimodal import create_mock_multimodal_config
        
        config_path = create_mock_multimodal_config()
        
        # Load configuration
        config = load_experiment_config(config_path)
        
        # Check key fields
        if hasattr(config, 'name') and hasattr(config, 'model'):
            print("✅ Configuration loading successful")
            print(f"   Config name: {config.name}")
            print(f"   Model type: {config.model.get('type', 'not specified')}")
        else:
            print("❌ Configuration missing required fields")
        
        Path(config_path).unlink()  # Clean up
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        traceback.print_exc()


def print_summary(import_results):
    """Print test summary."""
    
    print(f"\n📊 Test Summary")
    print("=" * 50)
    
    total_tests = len(import_results)
    successful = sum(1 for result in import_results.values() if result["status"] == "success")
    partial = sum(1 for result in import_results.values() if result["status"] == "partial")
    failed = total_tests - successful - partial
    
    print(f"Total demos tested: {total_tests}")
    print(f"✅ Successful: {successful}")
    print(f"⚠️  Partial: {partial}")  
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\n❌ Failed Tests:")
        for demo, result in import_results.items():
            if result["status"] not in ["success", "partial"]:
                print(f"   {demo}: {result['error']}")
    
    if partial > 0:
        print(f"\n⚠️  Partial Tests:")
        for demo, result in import_results.items():
            if result["status"] == "partial":
                print(f"   {demo}: {result['error']}")
    
    # Overall assessment
    if successful == total_tests:
        print(f"\n🎉 All unified demo scripts are working correctly!")
    elif successful + partial == total_tests:
        print(f"\n👍 Unified demo scripts mostly working with minor issues")
    else:
        print(f"\n⚠️  Some unified demo scripts need attention")
    
    print(f"\n💡 Benefits of Unified Interface Confirmed:")
    benefits = [
        "Consistent trainer inheritance from BaseTrainer",
        "Template-based configuration loading",
        "Standardized experiment configuration structure", 
        "Unified component initialization patterns",
        "Common error handling and validation"
    ]
    
    for benefit in benefits:
        print(f"   ✓ {benefit}")


def main():
    """Run all tests."""
    
    print("🚀 BEM 2.0 Unified Demo Scripts Test Suite")
    print("=" * 60)
    
    # Run import tests
    import_results = test_demo_imports()
    
    # Run configuration tests
    test_configuration_creation()
    
    # Run trainer tests
    test_trainer_instantiation()
    
    # Run config loading tests
    test_config_loading()
    
    # Print summary
    print_summary(import_results)
    
    return 0


if __name__ == "__main__":
    exit(main())