"""PyTest configuration and global fixtures.

This module is automatically loaded by pytest before running any tests.
It ensures that PyTorch's hardware-specific backends are disabled during
testing to prevent crashes on unsupported CI/CD runners.
"""

import torch


def pytest_configure() -> None:
    """Disable hardware-specific CPU backends during tests.
    
    Prevents 'RuntimeError: could not create a primitive' crashes on
    unsupported or heavily virtualized hardware architectures.
    """
    try:
        torch.backends.nnpack.set_flags(False)
    except Exception:
        pass
    try:
        torch.backends.mkldnn.enabled = False  # type: ignore[assignment]
    except Exception:
        pass
