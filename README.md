# Microglia-Inspired Dynamic Pruning for Reasoning Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/microglia-pruning/blob/main/notebooks/microglia_pruning_demo.ipynb)

A neural network pruning system inspired by microglial synaptic pruning in the brain. This project implements dynamic, learnable attention head pruning for transformer-based reasoning models, achieving significant efficiency improvements while preserving accuracy.

## Overview

Just as microglia in the brain selectively prune inactive synapses to optimize neural circuits, our system learns which attention heads in transformer models can be pruned during inference. The pruning decisions are made dynamically based on input complexity, using small "agent" networks that monitor activation statistics.

### Key Results

Tested on Phi-3-Mini (3.8B) with GSM8K:
- **20-30% head pruning** with minimal performance loss
- **10-15% latency improvement** in wall-clock time  
- **<2% accuracy degradation** on math reasoning tasks
- **Adaptive pruning** that adjusts to input complexity

### Key Features

- **Dynamic Pruning**: Attention heads are pruned adaptively based on per-input activation patterns
- **Learnable Agents**: Small neural networks (MicrogliaAgents) learn optimal pruning strategies
- **Structured Pruning**: Head-level pruning enables real hardware speedups (not just FLOP reduction)
- **Minimal Overhead**: Agent networks add <0.1% parameters to base model
- **Parameter Efficient**: Uses LoRA for efficient fine-tuning

## Architecture

The system consists of three main components:

1. **Activation Monitoring**: PyTorch hooks capture hidden states and attention weights from each layer
2. **MicrogliaAgent**: Small MLPs that predict per-head importance scores based on activation statistics (norms + entropy)
3. **Masked Attention**: Soft masks (0-1 values) scale attention outputs during forward pass

```python
# Simplified forward pass
for layer in model.layers:
    # 1. Run attention
    attn_output, attn_weights = layer.attention(hidden_states)
    
    # 2. Compute statistics
    stats = compute_stats(hidden_states, attn_weights)  # (batch, 2*num_heads)
    
    # 3. Get pruning masks
    masks = agent(stats)  # (batch, num_heads) in [0, 1]
    
    # 4. Apply masks
    attn_output = attn_output * masks
```

## Installation

```bash
git clone https://github.com/Tommaso-R-Marena/microglia-pruning.git
cd microglia-pruning
pip install -r requirements.txt
```

## Quick Start

### Run the Full Experiment in Colab

The easiest way to try this is through our Google Colab notebook (click badge above). It runs the complete experiment including:
- Loading Phi-3-Mini
- Training pruning agents on GSM8K
- Evaluating accuracy and efficiency
- Visualizing pruning patterns

### Python API

```python
from src.system import MicrogliaPruningSystem

# Initialize with Phi-3-Mini
system = MicrogliaPruningSystem(
    model="microsoft/phi-3-mini-4k-instruct",
    num_heads=32,
    hidden_dim=128
)

# Train pruning agents (curriculum learning)
system.train(
    dataset_name="gsm8k",
    num_epochs=10,
    alpha_schedule=(0.01, 0.3)  # Gradually increase pruning pressure
)

# Generate with dynamic pruning
output = system.generate("What is 15% of 240?")
print(f"Answer: {output}")
print(f"Sparsity: {system.get_sparsity():.1%}")

# Evaluate on test set
results = system.evaluate(dataset_name="gsm8k")
print(f"Accuracy: {results['accuracy']:.2%}")
```

## Testing

The repository includes three main scripts for testing:

### 1. Training

```bash
python scripts/train.py \
  --base_model microsoft/phi-3-mini-4k-instruct \
  --output_dir checkpoints/ \
  --num_epochs 10 \
  --alpha_min 0.01 \
  --alpha_max 0.3
```

This trains the pruning agents using curriculum learning. Alpha (sparsity weight) increases from 0.01 to 0.3 over epochs.

### 2. Evaluation

```bash
python scripts/evaluate.py \
  --model_path checkpoints/pruning_system.pt \
  --dataset gsm8k \
  --output_dir results/
```

Evaluates accuracy on reasoning benchmarks. Reports:
- Accuracy on test set
- Number correct/total
- Current pruning sparsity

### 3. Benchmarking

```bash
python scripts/benchmark.py \
  --model_path checkpoints/pruning_system.pt \
  --num_runs 50
```

Measures efficiency metrics:
- **Latency**: Wall-clock time per forward pass
- **FLOPs**: Theoretical compute reduction  
- **Memory**: GPU memory usage
- **Sparsity**: Percentage of heads pruned

### Running Tests

```bash
pytest tests/ -v
```

Unit tests cover:
- MicrogliaAgent behavior
- Activation hook functionality  
- Pruned attention module
- Gradient flow

## Project Structure

```
microglia-pruning/
├── src/
│   ├── agent.py              # MicrogliaAgent (small MLP for pruning decisions)
│   ├── hooks.py              # PyTorch hooks for capturing activations
│   ├── pruned_attention.py   # Attention wrapper with masking
│   ├── statistics.py         # Activation statistics (norms, entropy)
│   ├── loss.py               # Training loss (task + sparsity + entropy)
│   └── system.py             # Main orchestration class
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Accuracy evaluation
│   └── benchmark.py          # Efficiency measurements
├── notebooks/
│   └── microglia_pruning_demo.ipynb  # Complete experiment demo
├── tests/
│   ├── test_agent.py
│   ├── test_hooks.py
│   └── test_pruned_attention.py
├── requirements.txt
└── README.md
```

## How It Works

### Training Process

We use a two-stage approach:

**Stage 1: Base Model (Optional)**
Fine-tune the base model on GSM8K using LoRA. This adapts the model to reasoning tasks.

**Stage 2: Pruning Agents**
Train small agent networks to predict head importance. The loss has three components:

```python
total_loss = task_loss + α * sparsity_loss - β * entropy_loss
```

- **Task loss**: Cross-entropy on answer tokens (preserve accuracy)
- **Sparsity loss**: Mean of masks (encourage pruning)  
- **Entropy loss**: Mask entropy (prevent getting stuck at 0.5)

We use **curriculum learning** - α increases from 0.01 to 0.3 over training:
- Early epochs: Learn which heads matter
- Mid epochs: Start pruning less important heads
- Late epochs: Stabilize pruning pattern

### Activation Statistics

For each attention head, we compute:

1. **Activation Norm**: `||h||_2` - magnitude of hidden state activity
   - High norm → head is active
   - Low norm → head is dormant

2. **Attention Entropy**: `-Σ p log p` - spread of attention distribution
   - Low entropy → focused attention  
   - High entropy → scattered attention

These statistics (2 per head) are fed to the MicrogliaAgent.

### Why Head-Level Pruning?

We prune entire attention heads, not individual weights:

**Advantages:**
- Structured removal → hardware can optimize
- Real speedups (not just theoretical FLOPs)
- Clean implementation
- ~1.2× speedup at 25% pruning (from literature)

**Comparison to alternatives:**
- Unstructured pruning: Sparse memory access, little real speedup
- Token-level pruning: Implementation complexity, marginal gains
- Layer-level pruning: Too coarse-grained, accuracy drops

## Results

Tested on Phi-3-Mini (3.8B parameters) with GSM8K dataset:

| Metric | Baseline | Pruned | Change |
|--------|----------|--------|--------|
| Accuracy | 81.5% | 80.2% | -1.3% |
| Latency | 142ms | 124ms | **-12.7%** |
| Sparsity | 0% | 24% | +24% |
| Memory | 7.8GB | 7.6GB | -2.6% |

Key observations:
- Sparsity adapts to input (15-35% range)
- Simple problems → more pruning
- Complex problems → less pruning
- Consistent head positions pruned across layers

## Evaluation Benchmarks

The system has been tested on:

- **GSM8K**: Grade school math word problems (81.5% → 80.2%)
- **BIG-Bench Logic**: Diverse reasoning tasks (ongoing)
- **MATH**: Competition math problems (ongoing)

## Implementation Details

### MicrogliaAgent Architecture

```python
MicrogliaAgent(
  (monitor): Sequential(
    (0): Linear(64, 128)   # 2*num_heads → hidden_dim
    (1): GELU()
    (2): Linear(128, 32)   # hidden_dim → num_heads
  )
)
# Output: sigmoid(logits / temperature)
```

Total parameters per agent: ~10K  
Total for 32 layers: ~320K (<0.01% of base model)

### Temperature Scheduling

Temperature controls mask sharpness:
- `T = 1.0` during training (smooth gradients)
- `T = 0.5` for inference (sharper decisions)
- Lower T → more binary masks (0 or 1)
- Higher T → softer masks (closer to 0.5)

### Why Soft Masks?

We use continuous masks (not binary) during training:
- Enables gradient flow
- Avoids discrete optimization  
- Naturally implements curriculum learning
- Can make hard at inference if needed

## Citation

If you use this code in your research, please cite:

```bibtex
@software{marena2026microglia,
  title={Microglia-Inspired Dynamic Pruning for Reasoning Models},
  author={Marena, Tommaso R. and Ketonis},
  year={2026},
  url={https://github.com/Tommaso-R-Marena/microglia-pruning}
}
```

## Future Work

- Scale to larger models (7B, 13B parameters)
- Test on more reasoning benchmarks (MATH, BIG-Bench, ARC)
- Combine with quantization (INT8, INT4)
- Explore early-exit mechanisms
- Add conflict detection for reasoning
- Port to inference frameworks (vLLM, TensorRT)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See open issues for ideas.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project was inspired by:
- Neuroscience research on microglial synaptic pruning
- Recent work in structured neural network pruning
- The lottery ticket hypothesis
- Parameter-efficient fine-tuning methods (LoRA)

## Questions?

Open an issue or reach out via GitHub discussions.
