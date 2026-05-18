#!/usr/bin/env python3
"""
Docker Installation Validation Script

This script validates dependencies in a Docker image profile. Each profile
matches one of the Dockerfiles under docker/.
"""

import argparse
import sys
from pathlib import Path

try:
    from .validation_base import BaseInstallationValidator, find_project_root, tomllib
except ImportError:
    from validation_base import BaseInstallationValidator, find_project_root, tomllib


class DockerInstallationValidator(BaseInstallationValidator):
    """Validates installation in a Docker environment."""

    PROFILE_EXTRAS = {
        "basic": (),
        "sglang": ("sglang", "fa"),
        "vllm": ("vllm", "fa"),
        "full": ("sglang", "vllm", "fa", "te"),
        "dev": ("sglang", "vllm", "fa", "te"),
        "example": ("sglang", "fa"),
    }

    CUDA_SUBMODULES = {
        **BaseInstallationValidator.CUDA_SUBMODULES,
        "transformer-engine": ["transformer_engine.pytorch"],
    }

    CRITICAL_PACKAGES = {
        *BaseInstallationValidator.CRITICAL_PACKAGES,
        "transformer-engine",
    }

    def __init__(self, pyproject_path: Path | None = None, profile: str = "full"):
        super().__init__(pyproject_path)
        self.profile = profile

    def parse_pyproject(self):
        """Parse pyproject.toml and add profile-specific Docker extras."""
        super().parse_pyproject()

        if self.pyproject_path is None:
            return

        with open(self.pyproject_path, "rb") as f:
            data = tomllib.load(f)

        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        extras = self.PROFILE_EXTRAS[self.profile]
        print(f"Validating Docker profile '{self.profile}' with extras: {extras}")

        for extra_name in extras:
            for requirement in optional_deps.get(extra_name, []):
                self.add_requirement(requirement, required=True)

    def test_cuda_functionality(self):
        """Run CUDA functionality tests including Docker-specific packages."""
        super().test_cuda_functionality()

        if self.profile not in {"full", "dev"}:
            return

        print("\n=== Docker-Specific CUDA Tests ===")

        # Test transformer engine FP8 if available
        try:
            import torch

            if not torch.cuda.is_available():
                print("⚠ CUDA not available - skipping transformer engine tests")
                return

            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            # Set dimensions for a small test
            in_features = 128
            out_features = 256
            hidden_size = 64

            # Initialize model and inputs
            model = te.Linear(in_features, out_features, bias=True)
            inp = torch.randn(hidden_size, in_features, device="cuda")

            # Create an FP8 recipe
            fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

            # Enable autocasting for the forward pass
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(inp)

            loss = out.sum()
            loss.backward()
            print("✓ Transformer Engine FP8 operations")

        except ImportError:
            print("⚠ Transformer Engine not available - skipping FP8 tests")
        except Exception as e:
            print(f"⚠ Transformer Engine FP8 test failed: {e}")

    def get_validation_title(self) -> str:
        """Get the title for validation output."""
        return f"Docker Installation Validation ({self.profile})"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(DockerInstallationValidator.PROFILE_EXTRAS),
        default="full",
        help="Docker image profile to validate.",
    )
    args = parser.parse_args()

    try:
        project_root = find_project_root(Path(__file__))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    pyproject_path = project_root / "pyproject.toml"
    validator = DockerInstallationValidator(pyproject_path, profile=args.profile)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
