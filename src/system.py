"""Main pruning system that orchestrates training and inference."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import re
import os

from .agent import MicrogliaAgent
from .hooks import register_hooks, remove_hooks
from .pruned_attention import PrunedAttention
from .loss import compute_pruning_loss, get_alpha_schedule, compute_efficiency_metrics


class MicrogliaPruningSystem:
    """Complete system for microglia-inspired dynamic pruning.
    
    This orchestrates the entire pruning pipeline from model loading through
    training and evaluation. Based on biological inspiration from microglial
    synaptic pruning in the brain.
    """
    
    def __init__(self,
                 model: str or nn.Module,
                 num_heads: int = 32,
                 hidden_dim: int = 128,
                 temperature: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.num_heads = num_heads
        self.current_masks = {}  # Store masks from last forward pass
        
        print(f"Initializing MicrogliaPruningSystem on {device}...")
        
        # Load model and tokenizer
        if isinstance(model, str):
            print(f"Loading base model: {model}")
            
            # First load config to fix RoPE scaling issue
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            
            # Fix rope_scaling config if needed (Phi-3 issue)
            # The config sometimes has rope_scaling but missing 'type'
            # Valid types are: 'linear', 'longrope', or None
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                if isinstance(config.rope_scaling, dict) and 'type' not in config.rope_scaling:
                    # Just remove rope_scaling entirely - let model use default
                    config.rope_scaling = None
                    print("Removed invalid rope_scaling config (will use default)")
            
            # Now load model with fixed config
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Need this for hooks
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            
            # Phi models need padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.model = model
            self.tokenizer = None
        
        print(f"Model has {len(self.model.model.layers)} layers")
        
        # Create one agent per layer
        print(f"Initializing {len(self.model.model.layers)} pruning agents...")
        self.agents = nn.ModuleList([
            MicrogliaAgent(hidden_dim, num_heads, temperature)
            for _ in range(len(self.model.model.layers))
        ])
        self.agents.to(device)
        
        # Wrap attention layers
        self._wrap_attention_layers()
        
        # For monitoring during training
        self.activation_cache = {}
        self.training_history = []
        
        print("System initialized successfully!")
    
    def _wrap_attention_layers(self):
        """Replace standard attention with pruned attention."""
        print("Wrapping attention layers with pruning modules...")
        for idx, layer in enumerate(self.model.model.layers):
            original_attn = layer.self_attn
            layer.self_attn = PrunedAttention(
                original_attn, 
                self.agents[idx],
                hard_prune=False  # Use soft masks during training
            )
    
    def train(self,
             dataset_name: str = "gsm8k",
             num_epochs: int = 10,
             batch_size: int = 4,
             learning_rate: float = 1e-4,
             alpha_schedule: Tuple[float, float] = (0.01, 0.3),
             use_lora: bool = True,
             save_steps: int = 500):
        """Train the pruning agents on a reasoning dataset.
        
        We use a two-stage approach:
        1. Fine-tune base model on task (optional with LoRA)
        2. Train pruning agents while keeping model mostly frozen
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Load and prepare dataset
        print(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, "main")
        
        # Preprocess for training
        def preprocess_function(examples):
            # Format: question + answer for GSM8K
            prompts = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(examples['question'], examples['answer'])
            ]
            return self.tokenizer(
                prompts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
        
        print("Preprocessing dataset...")
        train_dataset = dataset['train'].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        # Apply LoRA if requested - makes training much more efficient
        if use_lora:
            print("Applying LoRA for parameter-efficient training...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Set up optimizer - only train agents initially
        print("\nSetting up optimizer for pruning agents...")
        optimizer = torch.optim.AdamW(
            self.agents.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training loop with curriculum learning
        self.model.train()
        alpha_min, alpha_max = alpha_schedule
        
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Alpha schedule: {alpha_min} -> {alpha_max}\n")
        
        for epoch in range(num_epochs):
            alpha = get_alpha_schedule(epoch, num_epochs, alpha_min, alpha_max)
            epoch_metrics = {'task_loss': 0.0, 'sparsity_loss': 0.0, 'total_loss': 0.0}
            
            print(f"\nEpoch {epoch+1}/{num_epochs} (alpha={alpha:.3f})")
            
            # Create dataloader
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch, labels=batch['input_ids'])
                task_loss = outputs.loss
                
                # Collect masks from all layers (stored during forward)
                all_masks = []
                for layer in self.model.model.layers:
                    if hasattr(layer.self_attn, 'last_masks'):
                        all_masks.append(layer.self_attn.last_masks)
                
                if all_masks:
                    masks = torch.cat(all_masks, dim=0)
                    
                    # Compute combined loss
                    loss_dict = compute_pruning_loss(task_loss, masks, alpha=alpha)
                    total_loss = loss_dict['total_loss']
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 1.0)
                    optimizer.step()
                    
                    # Update metrics
                    epoch_metrics['task_loss'] += loss_dict['task_loss']
                    epoch_metrics['sparsity_loss'] += loss_dict['sparsity_loss']
                    epoch_metrics['total_loss'] += total_loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{total_loss.item():.3f}",
                        'sparsity': f"{loss_dict['sparsity_loss']:.3f}"
                    })
                
                if step >= 100:  # Limit steps per epoch for demo
                    break
            
            # Epoch summary
            n_steps = min(len(train_loader), 100)
            avg_metrics = {k: v/n_steps for k, v in epoch_metrics.items()}
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Task Loss: {avg_metrics['task_loss']:.4f}")
            print(f"  Sparsity Loss: {avg_metrics['sparsity_loss']:.4f}")
            print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
            
            self.training_history.append(avg_metrics)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs):
        """Generate text with dynamic pruning enabled."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")
        
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with pruning
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_sparsity(self) -> float:
        """Calculate current average pruning sparsity across all layers."""
        if not self.current_masks:
            return 0.0
        
        all_masks = []
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'last_masks'):
                all_masks.append(layer.self_attn.last_masks)
        
        if not all_masks:
            return 0.0
        
        masks = torch.cat(all_masks, dim=0)
        metrics = compute_efficiency_metrics(masks)
        return metrics['sparsity']
    
    def evaluate(self, dataset_name: str = "gsm8k", split: str = "test", max_samples: int = 200) -> Dict:
        """Evaluate accuracy on a reasoning benchmark."""
        print(f"\nEvaluating on {dataset_name} ({split} split)...")
        
        dataset = load_dataset(dataset_name, "main", split=split)
        
        self.model.eval()
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc="Evaluating")
        
        with torch.no_grad():
            for example in progress_bar:
                # Generate answer
                prompt = f"Question: {example['question']}\nAnswer:"
                output = self.generate(prompt, max_new_tokens=256)
                
                # Extract answer (last number in generation)
                gold_answer = self._extract_answer(example['answer'])
                pred_answer = self._extract_answer(output)
                
                if pred_answer is not None and gold_answer is not None:
                    if abs(pred_answer - gold_answer) < 0.01:
                        correct += 1
                total += 1
                
                progress_bar.set_postfix({'accuracy': f"{correct/total:.1%}"})
        
        accuracy = correct / total if total > 0 else 0.0
        sparsity = self.get_sparsity()
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'sparsity': sparsity
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Correct: {correct}/{total}")
        print(f"  Sparsity: {sparsity:.1%}")
        
        return results
    
    def _extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text (for GSM8K)."""
        # Look for #### marker first (GSM8K format)
        if '####' in text:
            text = text.split('####')[1]
        
        # Find all numbers (including decimals)
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        
        if numbers:
            try:
                return float(numbers[-1])  # Last number is usually the answer
            except ValueError:
                return None
        return None
    
    def save(self, path: str):
        """Save the trained pruning system."""
        print(f"Saving model to {path}...")
        torch.save({
            'agents': self.agents.state_dict(),
            'model': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
            'training_history': self.training_history,
        }, path)
        print("Saved successfully!")
    
    def load(self, path: str):
        """Load a trained pruning system."""
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.agents.load_state_dict(checkpoint['agents'])
        if checkpoint.get('model') is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(checkpoint['model'])
        self.training_history = checkpoint.get('training_history', [])
        print("Loaded successfully!")
