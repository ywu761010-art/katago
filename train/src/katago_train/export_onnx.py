import argparse
import sys

import torch

from katago_train.dataset import REPO_ROOT
from katago_train.model import Net

MODELS_DIR = REPO_ROOT / "models"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--output", default=str(MODELS_DIR / "dev.onnx"))
    args = parser.parse_args()

    model = Net(args.size)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["policy", "value"],
        opset_version=18,
        external_data=False,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()