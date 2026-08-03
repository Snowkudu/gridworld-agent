import torch
from models.mlp import MLP

def test_mlp():
    model = MLP()
    example_states = torch.zeros(size=(32, 100), dtype=torch.float32)
    logits = model(example_states)
    assert logits.shape == (32, 4)
    assert logits.dtype == torch.float32

def test_mlp_accepts_grids():
    model = MLP()
    example_states = torch.zeros(size=(8, 10,10), dtype=torch.float32)
    logits = model(example_states)
    assert logits.shape == (8, 4)