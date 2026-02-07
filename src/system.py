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
import gc

from .agent import MicrogliaAgent
from .hooks import register_hooks, remove_hooks
from .pruned_attention import PrunedAttention
from .loss import compute_pruning_loss, get_alpha_schedule, compute_efficiency_metrics


class MicrogliaPruningSystem:
    """Complete system for microglia-inspired dynamic pruning."""
    
    def __init__(self,
                 model: str or nn.Module,
                 num_heads: int = 32,
                 hidden_dim: int = 128,
                 temperature: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.num_heads = num_heads
        self.current_masks = {}
        self.pruning_enabled = False
        
        print(f"Initializing MicrogliaPruningSystem on {device}...")
        
        if isinstance(model, str):
            print(f"Loading base model: {model}")
            
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            
            # Fix rope_scaling
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                if isinstance(config.rope_scaling, dict):
                    if 'type' not in config.rope_scaling:
                        config.rope_scaling = None
                        print("Disabled rope_scaling (missing type)")
                    elif config.rope_scaling.get('type') not in ['linear', 'dynamic', 'longrope']:
                        config.rope_scaling = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            
            # Fix Phi-3 EOS token issue
            if 'phi-3' in model.lower():
                print("Fixing Phi-3 EOS token issue...")
                self.tokenizer.eos_token = "<|end|>"
                self.tokenizer.pad_token = "<|end|>" 
                eos_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
                self.model.config.eos_token_id = eos_id
                self.model.config.pad_token_id = eos_id
                # Also set in generation config if it exists
                if hasattr(self.model, 'generation_config'):
                    self.model.generation_config.eos_token_id = eos_id
                    self.model.generation_config.pad_token_id = eos_id
                print(f"Set EOS token to '<|end|>' (ID: {eos_id})")
            else:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.model = model
            self.tokenizer = None
        
        print(f"Model has {len(self.model.model.layers)} layers")
        
        print(f"Initializing {len(self.model.model.layers)} pruning agents...")
        self.agents = nn.ModuleList([
            MicrogliaAgent(hidden_dim, num_heads, temperature)
            for _ in range(len(self.model.model.layers))
        ])
        self.agents.to(device)
        
        # Don't wrap initially
        self.wrapped = False
        self.lora_applied = False
        
        self.activation_cache = {}
        self.training_history = []
        
        print("System initialized successfully!")
        print("Note: Pruning is DISABLED until training starts")
    
    def _apply_lora(self):
        """Apply LoRA BEFORE wrapping attention."""
        if self.lora_applied:
            return
        
        print("Applying LoRA for parameter-efficient training...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.lora_applied = True
    
    def _wrap_attention_layers(self):
        """Replace standard attention with pruned attention."""
        if self.wrapped:
            return
        
        print("Wrapping attention layers with pruning modules...")
        for idx, layer in enumerate(self.model.model.layers):
            original_attn = layer.self_attn
            layer.self_attn = PrunedAttention(
                original_attn, 
                self.agents[idx],
                hard_prune=False
            )
            layer.self_attn.enable_pruning = False
        self.wrapped = True
    
    def _enable_pruning(self, enable: bool = True):
        """Enable or disable pruning in all layers."""
        self.pruning_enabled = enable
        if not self.wrapped:
            return
        
        for layer in self.model.model.layers:
            if isinstance(layer.self_attn, PrunedAttention):
                layer.self_attn.enable_pruning = enable
        
        print(f"Pruning {'ENABLED' if enable else 'DISABLED'}")
    
    def train(self,
             dataset_name: str = "gsm8k",
             num_epochs: int = 10,
             batch_size: int = 2,
             learning_rate: float = 1e-4,
             alpha_schedule: Tuple[float, float] = (0.01, 0.3),
             use_lora: bool = False,  # Disable LoRA for simplicity
             save_steps: int = 500):
        """Train the pruning agents on a reasoning dataset."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Apply LoRA first if requested (before wrapping)
        if use_lora and not self.lora_applied:
            self._apply_lora()
        
        # Then wrap attention
        if not self.wrapped:
            self._wrap_attention_layers()
        
        # Enable pruning for training
        self._enable_pruning(True)
        
        print(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, "main")
        
        print("Preprocessing dataset...")
        train_subset = dataset['train'].select(range(min(500, len(dataset['train']))))
        
        # Process in batches to avoid OOM
        processed_examples = []
        batch_size_preprocess = 100
        
        for i in range(0, len(train_subset), batch_size_preprocess):
            batch = train_subset[i:i+batch_size_preprocess]
            prompts = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(batch['question'], batch['answer'])
            ]
            encoded = self.tokenizer(
                prompts,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            for j in range(len(prompts)):
                processed_examples.append({
                    'input_ids': encoded['input_ids'][j],
                    'attention_mask': encoded['attention_mask'][j]
                })
            
            del encoded
            gc.collect()
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, examples):
                self.examples = examples
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                return self.examples[idx]
        
        train_dataset = SimpleDataset(processed_examples)
        print(f"Prepared {len(train_dataset)} training examples")
        
        print("\nSetting up optimizer for pruning agents...")
        optimizer = torch.optim.AdamW(
            self.agents.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.model.train()
        alpha_min, alpha_max = alpha_schedule
        
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Alpha schedule: {alpha_min} -> {alpha_max}\n")
        
        for epoch in range(num_epochs):
            alpha = get_alpha_schedule(epoch, num_epochs, alpha_min, alpha_max)
            epoch_metrics = {'task_loss': 0.0, 'sparsity_loss': 0.0, 'total_loss': 0.0}
            
            print(f"\nEpoch {epoch+1}/{num_epochs} (alpha={alpha:.3f})")
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch, labels=batch['input_ids'])
                task_loss = outputs.loss
                
                all_masks = []
                for layer in self.model.model.layers:
                    if hasattr(layer.self_attn, 'last_masks') and layer.self_attn.last_masks is not None:
                        all_masks.append(layer.self_attn.last_masks)
                
                if all_masks:
                    masks = torch.cat(all_masks, dim=0)
                    loss_dict = compute_pruning_loss(task_loss, masks, alpha=alpha)
                    total_loss = loss_dict['total_loss']
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_metrics['task_loss'] += loss_dict['task_loss']
                    epoch_metrics['sparsity_loss'] += loss_dict['sparsity_loss']
                    epoch_metrics['total_loss'] += total_loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f"{total_loss.item():.3f}",
                        'sparsity': f"{loss_dict['sparsity_loss']:.3f}"
                    })
                
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                
                if step >= 30:
                    break
            
            n_steps = min(len(train_loader), 30)
            avg_metrics = {k: v/n_steps for k, v in epoch_metrics.items()}
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Task Loss: {avg_metrics['task_loss']:.4f}")
            print(f"  Sparsity Loss: {avg_metrics['sparsity_loss']:.4f}")
            print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
            
            self.training_history.append(avg_metrics)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Disable pruning after training
        self._enable_pruning(False)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs):
        """Generate text WITHOUT pruning."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")
        
        # Ensure pruning is OFF
        self._enable_pruning(False)
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,  # Disable temperature for greedy
                top_p=None,
                use_cache=False,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def get_sparsity(self) -> float:
        """Calculate current average pruning sparsity."""
        if not self.wrapped:
            return 0.0
        
        all_masks = []
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'last_masks') and layer.self_attn.last_masks is not None:
                all_masks.append(layer.self_attn.last_masks)
        
        if not all_masks:
            return 0.0
        
        masks = torch.cat(all_masks, dim=0)
        metrics = compute_efficiency_metrics(masks)
        return metrics['sparsity']
    
    def evaluate(self, dataset_name: str = "gsm8k", split: str = "test", max_samples: int = 200) -> Dict:
        """Evaluate accuracy on a reasoning benchmark."""
        print(f"\nEvaluating on {dataset_name} ({split} split)...")
        
        self._enable_pruning(False)
        
        dataset = load_dataset(dataset_name, "main", split=split)
        
        self.model.eval()
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc="Evaluating")
        
        with torch.no_grad():
            for example in progress_bar:
                prompt = f"Question: {example['question']}\nAnswer:"
                output = self.generate(prompt, max_new_tokens=256)
                
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
        """Extract numerical answer from text."""
        if '####' in text:
            text = text.split('####')[1]
        
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None
    
    def save(self, path: str):
        """Save the trained pruning system."""
        print(f"Saving model to {path}...")
        torch.save({
            'agents': self.agents.state_dict(),
            'training_history': self.training_history,
        }, path)
        print("Saved successfully!")
    
    def load(self, path: str):
        """Load a trained pruning system."""
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.agents.load_state_dict(checkpoint['agents'])
        self.training_history = checkpoint.get('training_history', [])
        print("Loaded successfully!")
