from __future__ import annotations

import random

import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and Torch CUDA randomness."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_generator(seed: int) -> torch.Generator:
    """Create an independent seeded Torch generator."""
    return torch.Generator().manual_seed(seed)