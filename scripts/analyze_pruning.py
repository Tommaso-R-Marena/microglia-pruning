"""Script for analyzing and visualizing pruning patterns."""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.system import MicrogliaPruningSystem

def plot_pruning_heatmap(masks, output_path):
    """Plot a heatmap of pruning masks across layers and heads."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(masks, cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={'label': 'Keep Probability'})
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Dynamic Pruning Pattern (Heatmap)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

def plot_layer_distribution(masks, output_path):
    """Plot distribution of mask values per layer."""
    plt.figure(figsize=(12, 6))
    # Reshape for seaborn: [layer_idx, mask_value]
    num_layers, num_heads = masks.shape
    data = []
    for l in range(num_layers):
        for h in range(num_heads):
            data.append({"Layer": l, "Keep Probability": masks[l, h]})

    import pandas as pd
    df = pd.DataFrame(data)
    sns.violinplot(data=df, x="Layer", y="Keep Probability", inner="point")
    plt.title("Pruning Mask Distribution per Layer")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Layer distribution plot saved to {output_path}")

def plot_stat_correlations(stats, masks, output_path):
    """Analyze correlation between input stats and predicted masks."""
    # stats: (num_layers, num_heads, 4), masks: (num_layers, num_heads)
    plt.figure(figsize=(15, 10))
    stat_names = ["Mean Norm", "Std Norm", "Entropy", "Max Attn"]

    for i in range(4):
        plt.subplot(2, 2, i+1)
        x = stats[:, :, i].flatten()
        y = masks.flatten()
        plt.scatter(x, y, alpha=0.3)
        plt.xlabel(stat_names[i])
        plt.ylabel("Keep Probability")
        plt.title(f"Mask vs {stat_names[i]}")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Stat correlations plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze pruning patterns")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir", type=str, default="analysis/")
    parser.add_argument("--prompt", type=str, default="Calculate 15% of 240.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading system and model for analysis...")
    system = MicrogliaPruningSystem(model=args.base_model)
    system.load(args.model_path)
    system._enable_pruning(True)

    # Run a single generation to capture masks
    print(f"Running inference on prompt: '{args.prompt}'")
    _ = system.generate(args.prompt, max_new_tokens=10)

    # Collect masks and stats from layers
    all_masks = []
    all_stats = []
    layers = system.get_layers()

    # We need to register a hook or something to capture stats,
    # but the PrunedAttention already computes them.
    # Let's modify PrunedAttention to store the last stats for analysis.

    # Run a single generation to capture masks
    print(f"Running inference on prompt: '{args.prompt}'")
    _ = system.generate(args.prompt, max_new_tokens=10)

    for layer in layers:
        if hasattr(layer.self_attn, 'last_masks') and layer.self_attn.last_masks is not None:
            all_masks.append(layer.self_attn.last_masks[0].cpu().numpy())
        if hasattr(layer.self_attn, 'last_stats') and layer.self_attn.last_stats is not None:
            # last_stats shape is (batch, 4*num_heads)
            # reshape to (4, num_heads) for this head
            s = layer.self_attn.last_stats[0].cpu().numpy()
            num_heads = s.shape[0] // 4
            s = s.reshape(4, num_heads).T # (num_heads, 4)
            all_stats.append(s)

    if not all_masks:
        print("No masks captured. Ensure pruning is enabled.")
        return

    masks_matrix = np.stack(all_masks)
    stats_matrix = np.stack(all_stats) # (num_layers, num_heads, 4)

    # Generate visualizations
    plot_pruning_heatmap(masks_matrix, os.path.join(args.output_dir, "pruning_heatmap.png"))
    plot_layer_distribution(masks_matrix, os.path.join(args.output_dir, "layer_distribution.png"))
    plot_stat_correlations(stats_matrix, masks_matrix, os.path.join(args.output_dir, "stat_correlations.png"))

    # Statistics
    avg_sparsity = 1.0 - masks_matrix.mean()
    print(f"Average Sparsity for this input: {avg_sparsity:.1%}")

    # Save raw data
    np.save(os.path.join(args.output_dir, "masks_matrix.npy"), masks_matrix)
    print("Analysis complete.")

if __name__ == "__main__":
    main()
