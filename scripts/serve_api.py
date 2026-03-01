"""Run FastAPI serving app for microglia-pruning."""

import argparse
import os
import sys

import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(model_name=args.model, backend=args.backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
