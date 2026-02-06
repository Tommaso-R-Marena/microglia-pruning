"""Benchmarking script for efficiency measurements."""

import argparse
import os
import sys
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fvcore.nn import FlopCountAnalysis
from src.system import MicrogliaPruningSystem


def measure_latency(system, prompt, num_runs=50):
    """Measure average latency over multiple runs."""
    latencies = []
    
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = system.generate(prompt, max_new_tokens=256)
        latencies.append(time.time() - start)
    
    return {
        'mean_ms': sum(latencies) / len(latencies) * 1000,
        'min_ms': min(latencies) * 1000,
        'max_ms': max(latencies) * 1000,
    }


def measure_memory():
    """Measure GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0}


def main():
    parser = argparse.ArgumentParser(description="Benchmark pruning system efficiency")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of runs for latency measurement"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Microglia Pruning System Benchmarking")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Runs: {args.num_runs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Load system
    print("\nLoading model...")
    system = MicrogliaPruningSystem(model=args.base_model)
    system.load(args.model_path)
    
    # Test prompt
    test_prompt = "What is 15% of 240?"
    
    # Measure latency
    print("\nMeasuring latency...")
    latency_metrics = measure_latency(system, test_prompt, args.num_runs)
    
    # Measure memory
    print("Measuring memory usage...")
    memory_metrics = measure_memory()
    
    # Get sparsity
    sparsity = system.get_sparsity()
    
    # Compile results
    results = {
        'latency': latency_metrics,
        'memory': memory_metrics,
        'sparsity': sparsity,
        'config': {
            'num_runs': args.num_runs,
            'batch_size': args.batch_size,
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Average Latency: {latency_metrics['mean_ms']:.2f} ms")
    print(f"Min Latency: {latency_metrics['min_ms']:.2f} ms")
    print(f"Max Latency: {latency_metrics['max_ms']:.2f} ms")
    print(f"GPU Memory: {memory_metrics['allocated_mb']:.1f} MB")
    print(f"Sparsity: {sparsity:.1%} heads pruned")
    print("="*60)
    
    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    print(f"\nSaving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
