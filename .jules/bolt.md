## 2026-05-22 - [Autograd In-place Pitfall]
**Learning:** Using in-place operations (like `.mul_()`) on tensors that are required for gradient computation will break PyTorch's autograd. Specifically, if a tensor `A` is multiplied in-place by `B` where `B` requires grad, the backward pass for `B` cannot be computed because the original value of `A` was modified.
**Action:** Always use out-of-place operations (`A * B`) when the operation involves tensors that are part of a differentiable computational graph, especially in training-enabled paths.

## 2026-05-22 - [Optimizing Attention Stats]
**Learning:** `torch.special.entr` is significantly faster and more numerically stable than manual `- (p * p.log())` entropy calculation. `torch.amax` is faster than `torch.max(...)[0]` when indices are not needed. It can also be used for binary entropy by computing `entr(x) + entr(1-x)`.
**Action:** Prefer `torch.special` functions and `amax`/`amin` for performance-critical tensor reductions.

## 2026-05-22 - [Positional Encoding Caching]
**Learning:** In the Microglia pruning system, agents are per-layer but receive identical layer-positional encodings for every sample in a batch. Recomputing these encodings and performing a full linear projection for every sample in the batch is redundant.
**Action:** Cache the positional encoding in a buffer and project it once, then use PyTorch broadcasting to apply it to the batch. This reduced agent forward latency by ~20%.

## 2026-05-22 - [bitsandbytes/triton Integration Conflict]
**Learning:** Some versions of `bitsandbytes` (e.g., 0.43.1) attempt to import from `triton.ops`, which is missing in newer Triton versions (e.g., 3.6.0), causing a `ModuleNotFoundError`.
**Action:** Downgrade to `bitsandbytes==0.42.0` or ensure `triton` is uninstalled if only CPU support is needed for testing, to avoid breaking the PEFT/LoRA integration.
