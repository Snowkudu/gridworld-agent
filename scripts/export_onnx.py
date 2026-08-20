from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from models.checkpoint import build_model_from_config


def load_export_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict, dict, str]:
    """Load either a supervised model or the online model from a DQN checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dictionary.")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'config'.")

    if "model_state_dict" in checkpoint:
        model_config = config
        state_dict = checkpoint["model_state_dict"]
        checkpoint_type = "supervised"
    elif "online_state_dict" in checkpoint:
        model_config = config.get("cnn")
        if not isinstance(model_config, dict):
            raise ValueError(
                f"DQN checkpoint {checkpoint_path} does not contain 'config.cnn'."
            )
        state_dict = checkpoint["online_state_dict"]
        checkpoint_type = "dqn-online"
    else:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains neither "
            "'model_state_dict' nor 'online_state_dict'."
        )

    model = build_model_from_config(model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint, model_config, checkpoint_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")

    model, checkpoint, model_config, checkpoint_type = load_export_model(
        args.checkpoint,
        device,
    )

    input_ch = int(model_config["input_ch"])
    height = int(model_config.get("height", 10))
    width = int(model_config.get("width", 10))

    example_input = torch.zeros(
        1,
        input_ch,
        height,
        width,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(example_input)

    if logits.shape != (1, 4):
        raise ValueError(
            f"Expected model output shape (1, 4), got {tuple(logits.shape)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    report_dir = args.output.parent / "export_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    onnx_program = torch.onnx.export(
        model,
        (example_input,),
        input_names=["grid"],
        output_names=["logits"],
        dynamo=True,
        report=True,
        verify=True,
        artifacts_dir=report_dir,
    )

    onnx_program.save(
        args.output,
        external_data=False,
    )

    training_config = checkpoint["config"].get("training", {})
    inertia_config = training_config.get("inertia", {})
    manifest = {
        "schema_version": 1,
        "checkpoint_type": checkpoint_type,
        "model_file": args.output.name,
        "input": {
            "name": "grid",
            "shape": [1, input_ch, height, width],
            "dtype": "float32",
            "channels": (
                ["obstacles", "agent", "goal"] if input_ch == 3 else ["grid"]
            ),
        },
        "output": {
            "name": "logits",
            "shape": [1, 4],
            "actions": ["up", "down", "left", "right"],
        },
        "policy": {
            "inertia": {
                "enabled": bool(inertia_config.get("enabled", False)),
                "strength": float(inertia_config.get("strength", 0.0)),
            }
        },
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint type: {checkpoint_type}")
    print(f"Input shape: {tuple(example_input.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"ONNX model: {args.output}")
    print(f"Model manifest: {manifest_path}")
    print(f"Export report: {report_dir}")


if __name__ == "__main__":
    main()