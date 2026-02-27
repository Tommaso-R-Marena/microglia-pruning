"""Script for detailed error analysis of the pruning system."""

import argparse
import os
import sys
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.system import MicrogliaPruningSystem

def main():
    parser = argparse.ArgumentParser(description="Error analysis for pruning system")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--output_file", type=str, default="error_analysis.json")
    parser.add_argument("--max_samples", type=int, default=10)

    args = parser.parse_args()

    system = MicrogliaPruningSystem(model=args.base_model)
    system.load(args.model_path)

    dataset = load_dataset("gsm8k", "main", split="test")
    samples = dataset.select(range(min(args.max_samples, len(dataset))))

    results = []

    for example in tqdm(samples, desc="Analyzing Errors"):
        prompt = f"Question: {example['question']}\nAnswer:"
        gold_answer_text = example['answer']
        gold_num = system._extract_answer(gold_answer_text)

        # 1. Run without pruning (baseline)
        output_baseline = system.generate(prompt, use_pruning=False)
        baseline_num = system._extract_answer(output_baseline)

        # 2. Run with pruning
        output_pruned = system.generate(prompt, use_pruning=True)
        pruned_num = system._extract_answer(output_pruned)

        analysis = {
            "question": example["question"],
            "gold_answer": gold_answer_text,
            "baseline_output": output_baseline,
            "pruned_output": output_pruned,
            "baseline_correct": (baseline_num is not None and gold_num is not None and abs(baseline_num - gold_num) < 0.01),
            "pruned_correct": (pruned_num is not None and gold_num is not None and abs(pruned_num - gold_num) < 0.01),
            "sparsity": system.get_sparsity()
        }

        # Categorize
        if analysis["baseline_correct"] and not analysis["pruned_correct"]:
            analysis["category"] = "Pruning Regression"
        elif not analysis["baseline_correct"] and analysis["pruned_correct"]:
            analysis["category"] = "Pruning Improvement"
        elif not analysis["baseline_correct"] and not analysis["pruned_correct"]:
            analysis["category"] = "Base Model Error"
        else:
            analysis["category"] = "Success"

        results.append(analysis)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nError analysis saved to {args.output_file}")

    # Summary stats
    categories = [r["category"] for r in results]
    for cat in set(categories):
        count = categories.count(cat)
        print(f"  {cat}: {count} ({count/len(results):.1%})")

if __name__ == "__main__":
    main()
