# Running Complete Experiments

## Overview
The `complete_experiments.ipynb` notebook runs all publication-quality experiments in one session.

## Prerequisites
- Google Colab account (free tier works, but Pro recommended for A100 GPU)
- 3-4 hours of runtime (T4 GPU) or 2-3 hours (A100 GPU)
- Optional: Google Drive for auto-saving results

## What Gets Generated
1. **Figures** (300 DPI, publication-ready):
   - training_curves.png
   - accuracy_comparison.png
   - latency_distribution.png
   - ablation_heatmap.png
   - pareto_frontier.png
   - head_importance_heatmap.png

2. **Data Files**:
   - complete_experimental_results.json (all metrics)
   - ablation_results.json
   - pareto_results.json
   - partial_results.json (checkpoint)
   - summary_table.csv
   - summary_table.tex

3. **Checkpoints**:
   - trained_system.pt (main checkpoint)
   - system_checkpoint_*.pt (periodic saves every 10 min)

## Running Locally (Not Recommended)
If you must run locally instead of Colab:

```bash
git clone https://github.com/Tommaso-R-Marena/microglia-pruning.git
cd microglia-pruning
pip install -e .
jupyter notebook notebooks/complete_experiments.ipynb
```

Note: You'll need a GPU with ≥16GB VRAM. Colab is easier.

## Troubleshooting
### Notebook crashes mid-experiment
- Results are auto-saved to `partial_results.json`
- Re-run the notebook and use the built-in skip logic for completed experiments

### Out of memory
- Reduce `EVAL_MAX_SAMPLES` in Section 1
- Reduce `TRAIN_BATCH_SIZE`

### Slow performance
- Check GPU type (T4 vs A100) - runtime auto-adjusts
- Consider Colab Pro for A100 access

## Using Results in Your Paper
1. Download all figures from the Files panel (📁)
2. Load `summary_table.tex` in your LaTeX document:

```tex
\begin{table}[h]
\centering
\input{results/summary_table.tex}
\caption{Complete experimental results}
\label{tab:results}
\end{table}
```

3. Reference `complete_experimental_results.json` for exact numbers.

## Citation
See main README.md for the canonical BibTeX entry.
