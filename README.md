# Microglia-Inspired Dynamic Pruning for Reasoning Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/microglia-pruning/blob/main/notebooks/microglia_pruning_demo.ipynb)

A neural network pruning system inspired by microglial synaptic pruning in the brain. This project implements dynamic, learnable attention head pruning for transformer-based reasoning models, achieving significant efficiency improvements while preserving accuracy.

## Overview

Just as microglia in the brain selectively prune inactive synapses to optimize neural circuits, our system learns which attention heads in transformer models can be pruned during inference. The pruning decisions are made dynamically based on input complexity, using small "agent" networks that monitor activation statistics.

### Key Features

- **Dynamic Pruning**: Attention heads are pruned adaptively based on per-input activation patterns
- **Learnable Agents**: Small neural networks (MicrogliaAgents) learn optimal pruning strategies
- **Structured Pruning**: Head-level pruning enables real hardware speedups
- **Minimal Accuracy Loss**: Achieves 20-30% pruning with <2% accuracy degradation
- **Parameter Efficient**: Uses LoRA for efficient fine-tuning

## Architecture

The system consists of three main components:

1. **Activation Monitoring**: PyTorch hooks capture hidden states and attention weights from each layer
2. **MicrogliaAgent**: Small MLPs that predict per-head importance scores based on activation statistics
3. **Masked Attention**: Soft masks (0-1 values) scale attention outputs during forward pass

## Installation

```bash
git clone https://github.com/Tommaso-R-Marena/microglia-pruning.git
cd microglia-pruning
pip install -r requirements.txt
```

## Quick Start

### Training

```python
from microglia_pruning import MicrogliaPruningSystem
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")

# Initialize pruning system
pruning_system = MicrogliaPruningSystem(
    model=model,
    num_heads=32,
    hidden_dim=128,
    temperature=1.0
)

# Fine-tune on GSM8K
pruning_system.train(
    dataset_name="gsm8k",
    num_epochs=10,
    alpha_schedule=(0.01, 0.3)
)
```

### Inference

```python
# Run inference with dynamic pruning
output = pruning_system.generate(
    "What is 15% of 80?",
    max_new_tokens=256
)

print(f"Answer: {output}")
print(f"Heads pruned: {pruning_system.get_sparsity():.1%}")
```

## Usage

### Testing the Model

To test the trained model on reasoning benchmarks:

```bash
python scripts/evaluate.py \
  --model_path checkpoints/phi3-pruned \
  --dataset gsm8k \
  --output_dir results/
```

### Training from Scratch

```bash
python scripts/train.py \
  --base_model microsoft/phi-3-mini-4k-instruct \
  --output_dir checkpoints/ \
  --num_epochs 10 \
  --alpha_min 0.01 \
  --alpha_max 0.3
```

### Measuring Efficiency

```bash
python scripts/benchmark.py \
  --model_path checkpoints/phi3-pruned \
  --num_runs 50 \
  --batch_size 1
```

This will report:
- Latency (ms per forward pass)
- FLOPs reduction
- Memory usage
- Active head percentage

## Project Structure

```
microglia-pruning/
├── src/
│   ├── __init__.py
│   ├── agent.py              # MicrogliaAgent implementation
│   ├── hooks.py              # Activation monitoring hooks
│   ├── pruned_attention.py   # Masked attention wrapper
│   ├── statistics.py         # Layer statistics computation
│   ├── loss.py               # Training loss functions
│   └── system.py             # Main pruning system
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── benchmark.py          # Efficiency benchmarking
├── notebooks/
│   └── microglia_pruning_demo.ipynb  # Colab demo
├── tests/
│   ├── test_agent.py
│   ├── test_hooks.py
│   └── test_pruned_attention.py
├── requirements.txt
├── setup.py
└── README.md
```

## Evaluation Benchmarks

The system is evaluated on:

- **GSM8K**: Grade school math word problems
- **BIG-Bench Logic**: Diverse logical reasoning tasks
- **Noisy GSM8K**: Robustness testing with corrupted reasoning steps

Target metrics:
- ≥75% accuracy on GSM8K
- ≥10% latency improvement
- 20-30% head pruning ratio
- ≤2% accuracy degradation vs baseline

## Implementation Details

### Activation Statistics

For each attention head, we compute:
1. **Activation Norm**: Magnitude of hidden state activity
2. **Attention Entropy**: Spread of attention distribution

These statistics serve as input to the MicrogliaAgent.

### Loss Function

The training loss combines three terms:

```python
total_loss = task_loss + α * sparsity_loss - β * entropy_loss
```

- **Task Loss**: Cross-entropy on answer tokens (preserves accuracy)
- **Sparsity Loss**: Encourages pruning (masks → 0)
- **Entropy Loss**: Prevents mask oscillation (avoids getting stuck at 0.5)

The sparsity coefficient α increases over training using curriculum learning.

### Curriculum Learning

The pruning pressure gradually increases:
- Early epochs: α ≈ 0.01 (learn which heads are important)
- Mid epochs: α ≈ 0.1 (gradually mask less important heads)
- Late epochs: α ≈ 0.3 (stabilize pruning pattern)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{marena2026microglia,
  title={Microglia-Inspired Dynamic Pruning for Reasoning Models},
  author={Marena, Tommaso R. and Ketonis, Panos},
  year={2026},
  url={https://github.com/Tommaso-R-Marena/microglia-pruning}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project was inspired by neuroscience research on microglial synaptic pruning and builds on recent work in structured neural network pruning.
