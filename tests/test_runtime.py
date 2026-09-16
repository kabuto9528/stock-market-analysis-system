import random

import numpy as np
import torch

from src.runtime import set_random_seed


def test_set_random_seed_is_repeatable() -> None:
    set_random_seed(42)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    set_random_seed(42)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second
