"""日志和可复现实验所需的基础运行环境。"""

from __future__ import annotations

import logging
import os
import random


def configure_logging(level: int | str = logging.INFO) -> None:
    """为命令行脚本和服务层设置统一的基础日志格式。"""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_random_seed(seed: int = 42) -> None:
    """固定 Python、NumPy、PyTorch 与 CUDA 随机种子。"""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须为非负整数。")

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
