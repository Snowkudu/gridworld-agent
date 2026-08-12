from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.checkpoint import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")

    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device,
    )
    model.eval()

    config = checkpoint["config"]

    input_ch = int(config["input_ch"])
    height = int(config.get("height", 10))
    width = int(config.get("width", 10))

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

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input shape: {tuple(example_input.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"ONNX model: {args.output}")
    print(f"Export report: {report_dir}")


if __name__ == "__main__":
    main()
