from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from models.checkpoint import load_model_from_checkpoint

CHECKPOINT = Path("artifacts/p4_cnn/champion/checkpoint.pt")
ONNX_MODEL = Path("artifacts/p4_cnn/reports/champion/onnx/model.onnx")


def test_champion_onnx_matches_pytorch() -> None:
    model, checkpoint = load_model_from_checkpoint(
        CHECKPOINT,
        torch.device("cpu"),
    )
    model.eval()

    input_ch = int(checkpoint["config"]["input_ch"])

    x = torch.randn(
        1,
        input_ch,
        10,
        10,
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch_logits = model(x).cpu().numpy()

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    onnx_logits = session.run(
        ["logits"],
        {"grid": x.numpy()},
    )[0]

    np.testing.assert_allclose(
        onnx_logits,
        torch_logits,
        rtol=1e-5,
        atol=1e-5,
    )

    assert onnx_logits.shape == (1, 4)
    assert onnx_logits.argmax(axis=1).item() == torch_logits.argmax(axis=1).item()
