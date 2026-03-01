"""Export a HuggingFace CausalLM to ONNX."""

import argparse
import os
import sys

from transformers import AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.export import export_to_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--output", type=str, default="artifacts/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    path = export_to_onnx(model, args.output, opset=args.opset)
    print(f"Exported model to {path}")


if __name__ == "__main__":
    main()
