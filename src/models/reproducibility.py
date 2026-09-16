"""Python、NumPy 与 PyTorch 的随机种子和确定性设置。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .errors import ModelConfigurationError


def set_random_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """设置 Python、NumPy、PyTorch 与全部可用 CUDA 设备的随机种子。"""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ModelConfigurationError("随机种子必须为非负整数。")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
