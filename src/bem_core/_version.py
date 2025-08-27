"""
Version information for BEM (Block-wise Expert Modules).
This module provides version information for the package.
"""

__version__ = "2.0.0"
__version_info__ = tuple(int(num) for num in __version__.split('.'))

# Release metadata
__title__ = "bem"
__description__ = "Block-wise Expert Modules: Adaptive Neural Architecture for Generalist AI"
__url__ = "https://github.com/nathanrice/BEM"
__author__ = "Nathan Rice and BEM Research Team"
__author_email__ = "nathan@example.com"  # Update with actual email
__license__ = "MIT"
__copyright__ = "2024 Nathan Rice and BEM Research Team"

# Release information
__status__ = "Production/Stable"  # Development, Production/Stable
__keywords__ = [
    "machine-learning",
    "deep-learning", 
    "neural-networks",
    "expert-systems",
    "adaptive-ai",
    "routing",
    "transformers",
    "pytorch"
]

# Version history tracking
VERSION_HISTORY = {
    "2.0.0": {
        "date": "2024-12-XX",
        "description": "Mission-based architecture with comprehensive validation",
        "features": [
            "Agentic routing system with >90% accuracy",
            "Online learning with <2% catastrophic forgetting",
            "Multimodal integration with vision-text routing",
            "Constitutional safety with 31% violation reduction",
            "Performance optimization variants (15-40% improvements)"
        ],
        "breaking_changes": [
            "Restructured from monolithic to mission-based architecture",
            "New configuration format with YAML schemas",
            "Updated API for routing and expert selection"
        ]
    },
    "1.0.0": {
        "date": "2024-XX-XX", 
        "description": "Initial stable release with comprehensive BEM system",
        "features": [
            "Hierarchical routing (prefix→chunk→token)",
            "Multi-BEM composition with non-interference guarantees",
            "Retrieval-augmented expert selection",
            "Advanced training pipeline with statistical validation"
        ]
    }
}

def get_version():
    """Return the current version string."""
    return __version__

def get_version_info():
    """Return version information as a dictionary."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "title": __title__,
        "description": __description__,
        "url": __url__,
        "author": __author__,
        "license": __license__,
        "status": __status__,
        "keywords": __keywords__
    }

def print_version_info():
    """Print comprehensive version information."""
    info = get_version_info()
    
    print(f"{info['title']} v{info['version']}")
    print(f"{info['description']}")
    print(f"Author: {info['author']}")
    print(f"License: {info['license']}")
    print(f"Status: {info['status']}")
    print(f"URL: {info['url']}")
    
    if __version__ in VERSION_HISTORY:
        history = VERSION_HISTORY[__version__]
        print(f"\nRelease Notes for v{__version__}:")
        print(f"Date: {history.get('date', 'TBD')}")
        print(f"Description: {history['description']}")
        
        if history.get('features'):
            print("\nNew Features:")
            for feature in history['features']:
                print(f"  • {feature}")
        
        if history.get('breaking_changes'):
            print("\nBreaking Changes:")
            for change in history['breaking_changes']:
                print(f"  • {change}")

if __name__ == "__main__":
    print_version_info()