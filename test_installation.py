#!/usr/bin/env python3
"""
Test script to verify F1 Crash Predictor installation and dependencies.
"""

import sys


def test_imports():
    """Test if all required packages can be imported."""
    print("\n" + "="*60)
    print("Testing F1 Crash Predictor Installation")
    print("="*60 + "\n")
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'requests': 'requests',
        'imblearn': 'imbalanced-learn',
        'scipy': 'scipy',
        'tqdm': 'tqdm'
    }
    
    failed = []
    
    for module, package_name in packages.items():
        try:
            __import__(module)
            print(f"✓ {package_name:<25} OK")
        except ImportError as e:
            print(f"✗ {package_name:<25} FAILED: {str(e)}")
            failed.append(package_name)
    
    print("\n" + "-"*60)
    
    if not failed:
        print("✓ All dependencies installed successfully!")
        print("\nYou can now run the pipeline:")
        print("  python main.py --quick")
    else:
        print(f"✗ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\nPlease install missing packages:")
        print("  pip install " + " ".join(failed))
        return False
    
    print("\n" + "="*60 + "\n")
    return True


def test_project_structure():
    """Test if project files exist."""
    import os
    
    print("Checking project structure...")
    print("-"*60)
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'src/data_collection.py',
        'src/feature_engineering.py',
        'src/eda.py',
        'src/model_training.py',
        'src/utils.py',
        'src/__init__.py'
    ]
    
    missing = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    print("-"*60)
    
    if not missing:
        print("✓ All project files present")
    else:
        print(f"✗ {len(missing)} file(s) missing")
        return False
    
    print("\n" + "="*60 + "\n")
    return True


def test_module_imports():
    """Test if custom modules can be imported."""
    print("Testing custom module imports...")
    print("-"*60)
    
    modules = [
        'src.data_collection',
        'src.feature_engineering',
        'src.eda',
        'src.model_training',
        'src.utils'
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module} - ERROR: {str(e)}")
            failed.append(module)
    
    print("-"*60)
    
    if not failed:
        print("✓ All custom modules can be imported")
    else:
        print(f"✗ {len(failed)} module(s) failed to import")
        return False
    
    print("\n" + "="*60 + "\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("F1 CRASH PREDICTOR - INSTALLATION TEST")
    print("="*60)
    
    # Test Python version
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    if sys.version_info < (3, 8):
        print("\n✗ Python 3.8 or higher is required!")
        print(f"  Current version: {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    # Run tests
    test1 = test_imports()
    test2 = test_project_structure()
    test3 = test_module_imports() if test1 and test2 else False
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if test1 and test2 and test3:
        print("\nALL TESTS PASSED")
        print("\nF1 Crash Predictor is ready to use.")
        print("\nNext steps:")
        print("  1. Run quick test:  python main.py --quick --sessions 3")
        print("  2. Run full pipeline: python main.py")
        print("  3. See usage guide: cat USAGE.md")
        print("\n" + "="*60 + "\n")
        return True
    else:
        print("\nSOME TESTS FAILED")
        print("\nPlease fix the issues above and run this test again.")
        print("\n" + "="*60 + "\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

