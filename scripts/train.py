"""Training script for microglia pruning system."""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.system import MicrogliaPruningSystem
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train microglia pruning system")
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Training dataset name"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for agents"
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.01,
        help="Minimum sparsity weight"
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.3,
        help="Maximum sparsity weight"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for agents"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for agent sigmoid"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=30,
        help="Maximum steps per epoch"
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Microglia Pruning System Training")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Alpha schedule: [{args.alpha_min}, {args.alpha_max}]")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Initialize system
    print("\nInitializing pruning system...")
    system = MicrogliaPruningSystem(
        model=args.base_model,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature
    )
    
    # Train
    print("\nStarting training...")
    system.train(
        dataset_name=args.dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha_schedule=(args.alpha_min, args.alpha_max),
        use_lora=args.use_lora,
        max_steps_per_epoch=args.max_steps_per_epoch
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, "pruning_system.pt")
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    system.save(checkpoint_path)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
