"""Main pruning system that orchestrates training and inference."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from typing import Optional, Tuple

from .agent import MicrogliaAgent
from .hooks import register_hooks, remove_hooks
from .pruned_attention import PrunedAttention
from .loss import compute_pruning_loss, get_alpha_schedule


class MicrogliaPruningSystem:
    """Complete system for microglia-inspired dynamic pruning.
    
    This class handles:
    - Loading and preparing the base model
    - Initializing pruning agents for each layer
    - Training the pruning system
    - Running inference with dynamic pruning
    - Evaluating efficiency and accuracy
    
    Args:
        model: Base transformer model or model name
        num_heads: Number of attention heads per layer
        hidden_dim: Hidden dimension for MicrogliaAgent MLPs
        temperature: Temperature for sigmoid in agents
        device: Device to run on ('cuda' or 'cpu')
    """
    
    def __init__(self,
                 model: str or nn.Module,
                 num_heads: int = 32,
                 hidden_dim: int = 128,
                 temperature: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.num_heads = num_heads
        
        # Load model if string provided
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"  # Required for hooks
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            self.tokenizer = None
        
        # Initialize agents for each layer
        self.agents = nn.ModuleList([
            MicrogliaAgent(hidden_dim, num_heads, temperature)
            for _ in range(len(self.model.model.layers))
        ])
        
        # Move agents to device
        self.agents.to(device)
        
        # Wrap attention layers with pruning
        self._wrap_attention_layers()
        
        # Storage for monitoring
        self.activation_cache = {}
        self.mask_history = []
        
    def _wrap_attention_layers(self):
        """Replace attention layers with pruned versions."""
        for idx, layer in enumerate(self.model.model.layers):
            original_attn = layer.self_attn
            layer.self_attn = PrunedAttention(original_attn, self.agents[idx])
    
    def train(self,
             dataset_name: str = "gsm8k",
             num_epochs: int = 10,
             batch_size: int = 4,
             learning_rate: float = 1e-4,
             alpha_schedule: Tuple[float, float] = (0.01, 0.3),
             use_lora: bool = True):
        """Train the pruning system.
        
        Args:
            dataset_name: Name of HuggingFace dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for agents
            alpha_schedule: (min, max) for sparsity weight curriculum
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        # Load dataset
        dataset = load_dataset(dataset_name, "main", split="train")
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # Optimizer for agents only (freeze base model initially)
        optimizer = torch.optim.AdamW(self.agents.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        alpha_min, alpha_max = alpha_schedule
        
        for epoch in range(num_epochs):
            alpha = get_alpha_schedule(epoch, num_epochs, alpha_min, alpha_max)
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self._get_batches(dataset, batch_size)):
                # Forward pass
                outputs = self.model(**batch)
                
                # Compute loss
                # Note: This is simplified - actual implementation would need
                # to collect masks from all layers
                task_loss = outputs.loss
                
                # Backward and optimize
                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()
                
                epoch_loss += task_loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/(batch_idx+1):.4f}, Alpha: {alpha:.3f}")
    
    def _get_batches(self, dataset, batch_size: int):
        """Simple batching iterator (simplified for demo)."""
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            # Tokenize and prepare batch
            # This is simplified - real implementation would need proper collation
            yield batch
    
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs):
        """Generate text with dynamic pruning.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            generated_text: Generated output
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Provide tokenizer or use model name.")
        
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_sparsity(self) -> float:
        """Get current pruning sparsity (fraction of heads pruned)."""
        # This would need to track masks during forward pass
        # Placeholder implementation
        return 0.25
    
    def evaluate(self, dataset_name: str = "gsm8k", split: str = "test") -> dict:
        """Evaluate accuracy on a benchmark.
        
        Args:
            dataset_name: Name of evaluation dataset
            split: Dataset split to use
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        dataset = load_dataset(dataset_name, "main", split=split)
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for example in dataset:
                # Generate answer
                output = self.generate(example["question"], max_new_tokens=256)
                
                # Check correctness (simplified)
                # Real implementation would need proper answer extraction
                if self._check_answer(output, example["answer"]):
                    correct += 1
                total += 1
                
                if total >= 100:  # Limit for demo
                    break
        
        accuracy = correct / total
        return {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    def _check_answer(self, generated: str, gold: str) -> bool:
        """Check if generated answer matches gold (simplified)."""
        # This is a placeholder - real implementation would need
        # proper answer extraction and comparison
        return gold.lower() in generated.lower()
    
    def save(self, path: str):
        """Save trained system."""
        torch.save({
            'agents': self.agents.state_dict(),
            'model': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
        }, path)
    
    def load(self, path: str):
        """Load trained system."""
        checkpoint = torch.load(path)
        self.agents.load_state_dict(checkpoint['agents'])
        if checkpoint['model'] is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(checkpoint['model'])
