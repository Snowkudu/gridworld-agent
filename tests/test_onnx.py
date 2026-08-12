from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

from models.checkpoint import build_model_from_config, load_model_from_checkpoint


def test_onnx_export_matches_pytorch(tmp_path) -> None:
    config = {
        "model_type": "cnn",
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "dropout": 0.0,
        "fc_hidden": 64,
    }

    original_model = build_model_from_config(config)
    original_model.eval()

    checkpoint_path = tmp_path / "checkpoint.pt"
    onnx_path = tmp_path / "model.onnx"

    torch.save(
        {
            "config": config,
            "model_state_dict": original_model.state_dict(),
        },
        checkpoint_path,
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_onnx",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(onnx_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert onnx_path.exists()

    restored_model, _ = load_model_from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )

    x = torch.randn(
        1,
        3,
        10,
        10,
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch_logits = restored_model(x).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
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
