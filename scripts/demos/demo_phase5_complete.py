#!/usr/bin/env python3
"""
Phase 5 Complete BEM System Demonstration.

This script demonstrates the complete Phase 5 implementation with all advanced features:
- Banked Experts (MoE-style)
- Online Learning with Trust Monitors  
- Speculative Decoding
- Vector Quantization
- Counterfactual Routing

This completes the full TODO.md specification for the BEM research project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, List, Tuple

# Import all Phase 5 BEM components
from bem import (
    # Phase 5: Advanced features
    BankedExpertsModule,
    BankedExpertsConfig,
    create_banked_experts_module,
    create_default_banked_experts_config,
    
    OnlineLearningController, 
    OnlineLearningConfig,
    TrustStatus,
    create_online_learning_controller,
    create_default_online_learning_config,
    
    SpeculativeDecoder,
    SpeculativeDecodingConfig, 
    create_speculative_decoder,
    create_default_speculative_config,
    
    VectorQuantizer,
    VQConfig,
    create_vector_quantizer,
    create_default_vq_config,
    
    CounterfactualRoutingAnalyzer,
    CounterfactualConfig,
    ComponentType,
    create_counterfactual_analyzer,
    create_default_counterfactual_config,
    
    # Core components (from earlier phases)
    HierarchicalBEMModule,
    HierarchicalBEMConfig,
    create_hierarchical_bem,
    
    TelemetryCollector,
    create_telemetry_collector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedBEMSystem(nn.Module):
    """
    Complete Phase 5 BEM System integrating all advanced features.
    
    This class demonstrates how all Phase 5 components work together
    to create a state-of-the-art adaptive neural routing system.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 512,
        num_layers: int = 6
    ):
        super().__init__()
        
        # Base model components
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Core transformer layers (simplified)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        # Phase 5: Advanced Features Integration
        self._setup_phase5_components()
        
        logger.info("Advanced BEM System initialized with all Phase 5 features")
    
    def _setup_phase5_components(self):
        """Initialize all Phase 5 advanced features."""
        
        # 1. Banked Experts System
        banked_experts_config = create_default_banked_experts_config(
            num_experts=8,
            expert_rank=16,
            top_k=2,
            enable_batching=True
        )
        
        self.banked_experts = create_banked_experts_module(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            config=banked_experts_config
        )
        
        # 2. Vector Quantization
        vq_config = create_default_vq_config(
            codebook_size=256,
            code_dim=64,
            enable_residual=True,
            enable_episodic_memory=True
        )
        
        self.vector_quantizer = create_vector_quantizer(config=vq_config)
        self.vq_projection = nn.Linear(self.hidden_dim, vq_config.code_dim)
        self.vq_unprojection = nn.Linear(vq_config.code_dim, self.hidden_dim)
        
        # 3. Hierarchical BEM Controller (from earlier phases)
        hierarchical_config = HierarchicalBEMConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            chunk_size=32,
            enable_uncertainty=True,
            enable_token_routing=True
        )
        
        self.hierarchical_bem = create_hierarchical_bem(
            input_dim=self.hidden_dim,
            config=hierarchical_config
        )
        
        logger.info("All Phase 5 components initialized successfully")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_advanced_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete advanced BEM system.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            use_advanced_features: Whether to use Phase 5 advanced features
            
        Returns:
            Dictionary containing outputs and feature information
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding and positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Store intermediate outputs for analysis
        routing_info = {}
        vq_info = {}
        expert_info = {}
        
        if use_advanced_features:
            # Apply hierarchical BEM routing
            bem_result = self.hierarchical_bem(x)
            x = bem_result['output']
            routing_info = bem_result.get('routing_info', {})
            
            # Apply vector quantization to codes
            vq_input = self.vq_projection(x)
            vq_result = self.vector_quantizer(vq_input, training=self.training)
            vq_output = self.vq_unprojection(vq_result['quantized'])
            
            # Residual connection with quantized codes
            x = x + 0.1 * vq_output
            vq_info = {
                'quantized': vq_result['quantized'],
                'encoding_indices': vq_result['encoding_indices'],
                'commitment_loss': vq_result['commitment_loss'],
                'codebook_loss': vq_result['codebook_loss'],
                'metrics': vq_result['metrics']
            }
            
            # Apply banked experts
            expert_result, expert_routing = self.banked_experts(x, training=self.training)
            x = x + expert_result  # Residual connection
            expert_info = expert_routing
        
        # Core transformer processing
        for layer in self.transformer_layers:
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            if x.device != tgt_mask.device:
                tgt_mask = tgt_mask.to(x.device)
            
            x = layer(x, x, tgt_mask=tgt_mask)
        
        # Output projection
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'hidden_states': x,
            'routing_info': routing_info,
            'vq_info': vq_info,
            'expert_info': expert_info
        }


class Phase5Demonstrator:
    """Demonstrates all Phase 5 advanced features in action."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the advanced BEM system
        self.model = AdvancedBEMSystem(
            vocab_size=1000,
            hidden_dim=256,  # Smaller for demonstration
            num_layers=4
        ).to(self.device)
        
        # Create base model for speculative decoding comparison
        self.base_model = AdvancedBEMSystem(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3  # Smaller/faster for drafting
        ).to(self.device)
        
        # Mock tokenizer for demonstration
        self.tokenizer = self._create_mock_tokenizer()
        
        # Setup telemetry
        self.telemetry = create_telemetry_collector()
        
        # Initialize Phase 5 controllers
        self._setup_controllers()
        
        logger.info("Phase 5 Demonstrator initialized")
    
    def _create_mock_tokenizer(self):
        """Create a simple mock tokenizer for demonstration."""
        class MockTokenizer:
            def __init__(self, vocab_size=1000):
                self.vocab_size = vocab_size
                self.eos_token_id = 2
                
            def encode(self, text, return_tensors=None):
                # Simple character-based encoding
                tokens = [ord(c) % self.vocab_size for c in text[:50]]  # Limit length
                if return_tensors == 'pt':
                    return torch.tensor(tokens).unsqueeze(0)
                return tokens
                
            def decode(self, tokens):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                # Simple character decoding
                chars = [chr(t % 128 + 32) for t in tokens if t < 1000]  # Printable ASCII
                return ''.join(chars)
        
        return MockTokenizer()
    
    def _setup_controllers(self):
        """Setup all Phase 5 advanced controllers."""
        
        # 1. Online Learning Controller
        online_config = create_default_online_learning_config(
            base_learning_rate=1e-4,
            consolidation_frequency=500,
            enable_rollback=True
        )
        
        self.online_controller = create_online_learning_controller(
            model=self.model,
            config=online_config,
            telemetry_collector=self.telemetry
        )
        
        # Setup optimizer for online learning
        self.online_controller.setup_optimizer(
            torch.optim.AdamW,
            weight_decay=0.01
        )
        
        # 2. Speculative Decoding System
        speculative_config = create_default_speculative_config(
            draft_length=4,
            kl_threshold=0.1,
            enable_adaptive_drafting=True
        )
        
        self.speculative_decoder = create_speculative_decoder(
            base_model=self.base_model,
            bem_model=self.model,
            tokenizer=self.tokenizer,
            config=speculative_config,
            telemetry_collector=self.telemetry
        )
        
        # 3. Counterfactual Routing Analyzer
        def performance_evaluator(outputs, targets):
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Flatten for cross entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            return F.cross_entropy(logits_flat, targets_flat, ignore_index=-100).item()
        
        counterfactual_config = create_default_counterfactual_config(
            component_dropout_rate=0.1,
            analysis_frequency=50,
            enable_routing_optimization=True
        )
        
        self.counterfactual_analyzer = create_counterfactual_analyzer(
            model=self.model,
            performance_evaluator=performance_evaluator,
            config=counterfactual_config,
            telemetry_collector=self.telemetry
        )
        
        # Register components for counterfactual analysis
        self._register_counterfactual_components()
        
        logger.info("All Phase 5 controllers setup complete")
    
    def _register_counterfactual_components(self):
        """Register components for counterfactual analysis."""
        
        # Component dropout functions
        def banked_experts_dropout(module, is_active):
            if hasattr(module, 'banked_experts'):
                # Temporarily disable banked experts
                module.banked_experts.training = module.banked_experts.training and is_active
        
        def vq_dropout(module, is_active):
            if hasattr(module, 'vector_quantizer'):
                # Scale VQ contribution
                module.vector_quantizer.training = module.vector_quantizer.training and is_active
        
        def hierarchical_dropout(module, is_active):
            if hasattr(module, 'hierarchical_bem'):
                # Disable hierarchical routing
                module.hierarchical_bem.training = module.hierarchical_bem.training and is_active
        
        # Register components
        self.counterfactual_analyzer.register_component(
            "banked_experts", self.model, banked_experts_dropout
        )
        
        self.counterfactual_analyzer.register_component(
            "vector_quantizer", self.model, vq_dropout
        )
        
        self.counterfactual_analyzer.register_component(
            "hierarchical_bem", self.model, hierarchical_dropout
        )
    
    def demonstrate_banked_experts(self, batch_size: int = 4, seq_len: int = 32):
        """Demonstrate banked experts functionality."""
        logger.info("=== Demonstrating Banked Experts System ===")
        
        # Generate sample input
        input_ids = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
        
        # Forward pass with banked experts
        self.model.train()
        result = self.model(input_ids, use_advanced_features=True)
        expert_info = result['expert_info']
        
        print(f"Banked Experts Results:")
        print(f"  Load balancing loss: {expert_info['load_balancing_loss']:.4f}")
        print(f"  Active experts: {expert_info['num_active_experts']}")
        print(f"  Routing entropy: {expert_info['routing_entropy']:.4f}")
        
        # Get expert utilization statistics
        if hasattr(self.model.banked_experts, 'get_expert_statistics'):
            stats = self.model.banked_experts.get_expert_statistics()
            util = stats['utilization']
            print(f"  Expert utilization:")
            print(f"    Entropy: {util.entropy:.4f}")
            print(f"    Gini coefficient: {util.gini_coefficient:.4f}")
            print(f"    Active experts: {util.active_experts}/8")
    
    def demonstrate_online_learning(self, num_steps: int = 20):
        """Demonstrate online learning with trust monitors."""
        logger.info("=== Demonstrating Online Learning System ===")
        
        batch_size, seq_len = 2, 16
        
        for step in range(num_steps):
            # Generate sample data
            input_ids = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
            targets = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, use_advanced_features=True)
            
            # Compute loss
            logits = outputs['logits']
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            
            # Backward pass
            loss.backward()
            
            # Online learning step with safety monitoring
            result = self.online_controller.step(loss)
            
            if step % 5 == 0:
                status = self.online_controller.get_status_summary()
                print(f"Step {step}:")
                print(f"  Trust status: {status['trust_status']}")
                print(f"  Trust score: {status['trust_score']:.3f}")
                print(f"  Learning rate: {status['current_lr']:.2e}")
                print(f"  Step taken: {result['step_taken']}")
                
                if result['consolidation_triggered']:
                    print("  >>> Consolidation triggered")
                if result['rollback_triggered']:
                    print("  >>> Rollback triggered")
    
    def demonstrate_speculative_decoding(self, prompt: str = "The future of AI is"):
        """Demonstrate speculative decoding for fast generation."""
        logger.info("=== Demonstrating Speculative Decoding System ===")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        print(f"Input prompt: '{prompt}'")
        print(f"Input tokens shape: {input_ids.shape}")
        
        # Generate with speculative decoding
        start_time = time.time()
        result = self.speculative_decoder.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8
        )
        generation_time = time.time() - start_time
        
        # Display results
        metrics = result['metrics']
        print(f"Generated text: '{result['generated_text']}'")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Speculative Decoding Metrics:")
        print(f"  Acceptance rate: {metrics.acceptance_rate:.3f}")
        print(f"  Average draft length: {metrics.average_draft_length:.1f}")
        print(f"  Total tokens/sec: {metrics.total_tokens_per_second:.1f}")
        print(f"  Net speedup: {metrics.net_speedup:.2f}x")
        
        # Get decoder statistics
        stats = self.speculative_decoder.get_statistics()
        print(f"  Overall acceptance rate: {stats['overall_acceptance_rate']:.3f}")
    
    def demonstrate_vector_quantization(self, batch_size: int = 4, seq_len: int = 32):
        """Demonstrate vector quantization system."""
        logger.info("=== Demonstrating Vector Quantization System ===")
        
        # Generate sample input
        input_ids = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
        
        # Forward pass to get VQ information
        result = self.model(input_ids, use_advanced_features=True)
        vq_info = result['vq_info']
        
        if vq_info:
            metrics = vq_info['metrics']
            print(f"Vector Quantization Results:")
            print(f"  Codebook utilization: {metrics.codebook_utilization:.3f}")
            print(f"  Quantization error: {metrics.average_quantization_error:.4f}")
            print(f"  Perplexity: {metrics.perplexity:.2f}")
            print(f"  Dead codes: {metrics.dead_codes}")
            print(f"  Memory efficiency: {metrics.memory_efficiency:.3f}")
            
            # Get detailed codebook analysis
            if hasattr(self.model.vector_quantizer, 'get_codebook_analysis'):
                analysis = self.model.vector_quantizer.get_codebook_analysis()
                print(f"  Active codes: {analysis['active_codes']}/{analysis['codebook_size']}")
                print(f"  Total usage: {analysis['total_usage']}")
    
    def demonstrate_counterfactual_routing(self, num_steps: int = 15):
        """Demonstrate counterfactual routing analysis."""
        logger.info("=== Demonstrating Counterfactual Routing Analysis ===")
        
        batch_size, seq_len = 2, 24
        
        for step in range(num_steps):
            # Generate sample data
            input_ids = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
            targets = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
            
            # Counterfactual analysis step
            result = self.counterfactual_analyzer.step(input_ids, targets)
            
            if result['analysis_performed'] and step % 5 == 0:
                summary = self.counterfactual_analyzer.get_analysis_summary()
                print(f"\nStep {step} - Counterfactual Analysis:")
                print(f"  Routing efficiency: {summary['routing_efficiency']:.3f}")
                print(f"  Credit variance: {summary['credit_variance']:.3f}")
                print(f"  Ablation consistency: {summary['ablation_consistency']:.3f}")
                
                if summary['top_components']:
                    print(f"  Component importance ranking:")
                    for i, (name, importance, (ci_low, ci_high)) in enumerate(summary['top_components']):
                        print(f"    {i+1}. {name}: {importance:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Run systematic ablation at the end
        print(f"\nRunning systematic ablation study...")
        input_ids = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
        targets = torch.randint(0, self.model.vocab_size, (batch_size, seq_len)).to(self.device)
        
        ablation_results = self.counterfactual_analyzer.run_systematic_ablation(input_ids, targets)
        
        print("Systematic Ablation Results:")
        for component, result in ablation_results.items():
            print(f"  {component}: {result['mean_impact']:.4f} ± {result['std_impact']:.4f}")
    
    def run_complete_demonstration(self):
        """Run complete demonstration of all Phase 5 features."""
        logger.info("🚀 Starting Complete Phase 5 BEM System Demonstration")
        
        print("=" * 80)
        print("PHASE 5 BEM SYSTEM - COMPLETE DEMONSTRATION")
        print("Advanced Features: Banked Experts | Online Learning | Speculative Decoding")
        print("                  Vector Quantization | Counterfactual Routing")
        print("=" * 80)
        
        try:
            # 1. Banked Experts
            self.demonstrate_banked_experts()
            print("\n" + "-" * 80 + "\n")
            
            # 2. Online Learning
            self.demonstrate_online_learning(num_steps=15)
            print("\n" + "-" * 80 + "\n")
            
            # 3. Speculative Decoding  
            self.demonstrate_speculative_decoding("Hello world, this is a test of")
            print("\n" + "-" * 80 + "\n")
            
            # 4. Vector Quantization
            self.demonstrate_vector_quantization()
            print("\n" + "-" * 80 + "\n")
            
            # 5. Counterfactual Routing
            self.demonstrate_counterfactual_routing(num_steps=12)
            print("\n" + "-" * 80 + "\n")
            
            # Summary
            print("✅ Phase 5 Complete BEM System Demonstration Successful!")
            print("\nAll advanced features demonstrated:")
            print("  ✓ Banked Experts: MoE-style expert routing with load balancing")
            print("  ✓ Online Learning: Safe adaptation with trust monitors and rollback")
            print("  ✓ Speculative Decoding: Performance-neutral BEM application")
            print("  ✓ Vector Quantization: Discrete codes with episodic memory")
            print("  ✓ Counterfactual Routing: Component importance analysis")
            print("\n🎯 BEM Research Project Phase 5 - COMPLETE")
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise


def main():
    """Main demonstration function."""
    print("Initializing Phase 5 Complete BEM System...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create and run demonstration
    demonstrator = Phase5Demonstrator()
    demonstrator.run_complete_demonstration()


if __name__ == "__main__":
    main()