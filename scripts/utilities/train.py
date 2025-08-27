"""
BEM v1.1 Training Script

Main training script for BEM-v1.1-stable according to TODO.md specifications.

Usage:
    python train.py --exp experiments/B1_v11.yml --seeds 1,2,3,4,5 --log_dir logs/B1 --cache_safe on --sticky on --attn_bias on

Features:
- E1+E3+E4 architecture training
- Spectral + Frobenius governance
- Mixed precision (bf16/fp16)
- Cache metrics logging
- 5-seed experimental protocol
- 24GB VRAM budget compliance
"""

import argparse
import os
import sys
import yaml
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Add bem package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bem.models import create_bem_v11_model
from bem.training import BEMv11Trainer, BEMv11TrainingConfig
from bem.training.cache_metrics import benchmark_bem_model
from bem.retrieval_features import create_retrieval_feature_extractor

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


def setup_logging(output_dir: str, seed: int):
    """Setup logging for training run."""
    log_file = os.path.join(output_dir, f'training_seed_{seed}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['model', 'data', 'training', 'bem']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    return config


def create_dataset(data_config: Dict[str, Any]) -> Dataset:
    """Create training dataset from configuration."""
    if 'huggingface_dataset' in data_config:
        # Load from HuggingFace datasets
        dataset_name = data_config['huggingface_dataset']
        dataset = load_dataset(dataset_name, split=data_config.get('split', 'train'))
    elif 'jsonl_path' in data_config:
        # Load from local JSONL file
        jsonl_path = data_config['jsonl_path']
        with open(jsonl_path, 'r') as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)
    else:
        raise ValueError("Must specify either 'huggingface_dataset' or 'jsonl_path' in data config")
    
    return dataset


def create_data_collator(tokenizer, max_length: int = 4096):
    """Create data collator for BEM training."""
    
    def collate_fn(batch):
        # Extract text data
        texts = []
        labels = []
        retrieval_contexts = []
        
        for item in batch:
            if 'input' in item and 'output' in item:
                # Instruction format
                text = f"{item['input']} {item['output']}"
                label = item['output']
            elif 'text' in item:
                # Plain text format
                text = item['text']
                label = text
            else:
                raise ValueError("Unknown data format")
            
            texts.append(text)
            labels.append(label)
            
            # Retrieval context if available
            retrieval_contexts.append(item.get('retrieval_context', {}))
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Tokenize labels
        label_tokenized = tokenizer(
            labels,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': label_tokenized['input_ids'],
            'retrieval_context': retrieval_contexts if any(retrieval_contexts) else None
        }
    
    return collate_fn


def run_training(
    config: Dict[str, Any],
    seed: int,
    log_dir: str,
    cache_safe: bool = True,
    sticky_routing: bool = True,
    attention_bias: bool = True
) -> Dict[str, Any]:
    """Run single training with specified seed."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting BEM v1.1 training with seed {seed}")
    
    # Create output directory
    run_dir = os.path.join(log_dir, f'seed_{seed}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer_name = config['model']['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"📝 Loaded tokenizer: {tokenizer_name}")
    
    # Create BEM model
    model_name = config['model']['base_model']
    retrieval_index = config['model']['retrieval_index']
    encoder_name = config['model'].get('encoder', 'sentence-transformers/all-MiniLM-L6-v2')
    
    logger.info(f"🏗️  Creating BEM v1.1 model from {model_name}")
    
    bem_model = create_bem_v11_model(
        model_name_or_path=model_name,
        retrieval_index_path=retrieval_index,
        encoder_name_or_path=encoder_name,
        rank_schedule=config['bem'].get('rank_schedule', [2, 4, 8, 8, 8, 4, 2]),
        num_experts=config['bem'].get('num_experts', 2),
        chunk_size=config['bem'].get('chunk_size', 128),
        hysteresis_tau=config['bem'].get('hysteresis_tau', 0.7),
        attention_bias_enabled=attention_bias,
        governance_config={
            'max_singular_value': config['bem'].get('max_singular_value', 1.0),
            'fro_budget': config['bem'].get('fro_budget', 1.0),
            'decorrelation_weight': config['bem'].get('decorrelation_weight', 0.01),
            'flip_penalty_weight': config['bem'].get('flip_penalty_weight', 0.1)
        }
    )
    
    # Validate cache safety
    if cache_safe:
        cache_report = bem_model.get_cache_safety_report()
        if not cache_report['cache_safe']:
            raise ValueError("❌ Cache safety validation failed!")
        logger.info("✅ Cache safety validated")
    
    # Create datasets
    train_dataset = create_dataset(config['data']['train'])
    eval_dataset = create_dataset(config['data']['eval']) if 'eval' in config['data'] else None
    
    logger.info(f"📊 Train dataset: {len(train_dataset)} examples")
    if eval_dataset:
        logger.info(f"📊 Eval dataset: {len(eval_dataset)} examples")
    
    # Create BEM training configuration
    bem_config = BEMv11TrainingConfig(
        learning_rate=config['training'].get('learning_rate', 2e-4),
        weight_decay=config['training'].get('weight_decay', 0.01),
        warmup_ratio=config['training'].get('warmup_ratio', 0.05),
        num_epochs=config['training'].get('num_epochs', 3),
        batch_size=config['training'].get('batch_size', 4),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 8),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        fp16=config['training'].get('fp16', False),
        bf16=config['training'].get('bf16', True),
        governance_weight=config['bem'].get('governance_weight', 0.1),
        entropy_regularization=config['bem'].get('entropy_regularization', 0.01),
        chunk_size=config['bem'].get('chunk_size', 128),
        output_dir=run_dir,
        run_name=f"bem-v11-seed-{seed}"
    )
    
    # Create trainer
    trainer = BEMv11Trainer(
        model=bem_model,
        args=None,  # Will be created in BEMv11Trainer
        bem_config=bem_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=create_data_collator(tokenizer, config['training'].get('max_length', 4096))
    )
    
    logger.info("🔧 Created BEM v1.1 trainer")
    
    # Run training
    logger.info("🎯 Starting training...")
    training_output = trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(run_dir)
    
    # Run final evaluation
    logger.info("📊 Running final evaluation...")
    eval_results = trainer.evaluate() if eval_dataset else {}
    
    # Benchmark performance
    logger.info("⚡ Benchmarking performance...")
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,  # For accurate benchmarking
            collate_fn=create_data_collator(tokenizer)
        )
        benchmark_results = benchmark_bem_model(
            bem_model, eval_dataloader, num_batches=min(10, len(eval_dataset))
        )
    else:
        benchmark_results = {}
    
    # Compile results
    final_results = {
        'seed': seed,
        'config': config,
        'training_output': training_output,
        'eval_results': eval_results,
        'benchmark_results': benchmark_results,
        'cache_safety_report': bem_model.get_cache_safety_report(),
        'model_path': run_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"✅ Training completed for seed {seed}")
    
    return final_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='BEM v1.1 Training Script')
    
    # Required arguments
    parser.add_argument('--exp', type=str, required=True,
                       help='Path to experiment configuration YAML')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5',
                       help='Comma-separated list of seeds')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory for training logs and outputs')
    
    # BEM-specific flags
    parser.add_argument('--cache_safe', type=str, choices=['on', 'off'], default='on',
                       help='Enable cache safety validation')
    parser.add_argument('--sticky', type=str, choices=['on', 'off'], default='on',
                       help='Enable chunk-sticky routing')
    parser.add_argument('--attn_bias', type=str, choices=['on', 'off'], default='on',
                       help='Enable attention bias')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for training')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Load experiment configuration
    try:
        config = load_experiment_config(args.exp)
    except Exception as e:
        print(f"❌ Error loading experiment config: {e}")
        return 1
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Convert string flags to boolean
    cache_safe = args.cache_safe == 'on'
    sticky_routing = args.sticky == 'on'
    attention_bias = args.attn_bias == 'on'
    
    print(f"🧪 BEM v1.1 Training Campaign")
    print(f"   Experiment: {args.exp}")
    print(f"   Seeds: {seeds}")
    print(f"   Cache Safe: {cache_safe}")
    print(f"   Sticky Routing: {sticky_routing}")
    print(f"   Attention Bias: {attention_bias}")
    print(f"   Output: {args.log_dir}")
    print()
    
    # Run training for each seed
    all_results = []
    
    for seed in seeds:
        print(f"🔄 Running training with seed {seed}")
        
        try:
            # Setup logging for this seed
            setup_logging(args.log_dir, seed)
            
            # Run training
            results = run_training(
                config=config,
                seed=seed,
                log_dir=args.log_dir,
                cache_safe=cache_safe,
                sticky_routing=sticky_routing,
                attention_bias=attention_bias
            )
            
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Training failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregated results
    summary_results = {
        'experiment_config': args.exp,
        'seeds_completed': [r['seed'] for r in all_results],
        'total_seeds': len(seeds),
        'success_rate': len(all_results) / len(seeds),
        'individual_results': all_results,
        'campaign_timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(args.log_dir, 'campaign_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"\n🎉 Training campaign completed!")
    print(f"   Successful runs: {len(all_results)}/{len(seeds)}")
    print(f"   Summary saved to: {summary_path}")
    
    if len(all_results) == 0:
        print(f"❌ All training runs failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())