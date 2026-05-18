#!/usr/bin/env python3
"""
Dynamic Installation Validation Script

This script validates that all dependencies listed in pyproject.toml are properly
installed with correct versions and that CUDA extensions are functional.
"""

import argparse
import sys
from pathlib import Path

try:
    from .validation_base import BaseInstallationValidator, find_project_root
except ImportError:
    from validation_base import BaseInstallationValidator, find_project_root


class DynamicInstallationValidator(BaseInstallationValidator):
    """Validates installation based on pyproject.toml dependencies."""

    def get_validation_title(self) -> str:
        """Get the title for validation output."""
        return "Installation Validation"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    try:
        project_root = find_project_root(Path(__file__))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    pyproject_path = project_root / "pyproject.toml"
    validator = DynamicInstallationValidator(pyproject_path)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
