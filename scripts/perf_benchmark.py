import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Simple throughput benchmark for HF vs vLLM backends."""

import argparse
import time

from src.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()

    engine = InferenceEngine(model_name=args.model, backend=args.backend)
    prompts = [f"Benchmark prompt {i}" for i in range(args.requests)]

    start = time.perf_counter()
    for prompt in prompts:
        engine.generate(prompt)
    elapsed = time.perf_counter() - start

    rps = args.requests / elapsed if elapsed > 0 else 0.0
    print(f"backend={args.backend} requests={args.requests} elapsed={elapsed:.2f}s rps={rps:.2f}")


if __name__ == "__main__":
    main()
