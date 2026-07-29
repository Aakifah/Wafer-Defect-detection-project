"""Global configuration and initialization for the source package.

disable NNpack and MKLDNN as they will always fail the pipeline otherwise
"""

import torch

try:
    torch.backends.nnpack.set_flags(False)
except Exception:
    pass

try:
    setattr(torch.backends.mkldnn, "enabled", False)
except Exception:
    pass
