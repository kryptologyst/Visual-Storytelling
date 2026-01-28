#!/usr/bin/env python3
"""
Pre-commit configuration setup

This script sets up pre-commit hooks for code quality.
"""

import subprocess
import sys
from pathlib import Path


def install_pre_commit():
    """Install pre-commit hooks."""
    try:
        # Install pre-commit
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        print("✓ Pre-commit installed")
        
        # Install hooks
        subprocess.run(["pre-commit", "install"], check=True)
        print("✓ Pre-commit hooks installed")
        
        # Run on all files
        subprocess.run(["pre-commit", "run", "--all-files"], check=True)
        print("✓ Pre-commit hooks run successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error setting up pre-commit: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    print("Setting up pre-commit hooks...")
    
    if install_pre_commit():
        print("\n✓ Pre-commit setup completed successfully!")
        print("Code quality hooks are now active.")
    else:
        print("\n✗ Pre-commit setup failed.")
        print("Please install manually: pip install pre-commit && pre-commit install")


if __name__ == "__main__":
    main()
