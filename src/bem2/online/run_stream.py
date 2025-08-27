#!/usr/bin/env python3
"""
BEM 2.0 Online Learning Stream Runner

This script demonstrates how to run the complete BEM 2.0 online learning system
with live feedback streams, implementing the workflow from TODO.md.

Usage:
    python run_stream.py --config config.json
    python run_stream.py --warmup-from /path/to/ar1_checkpoint.pt
    python run_stream.py --soak-test --duration-hours 24
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import time
from datetime import datetime, timedelta

from .online_learner import OnlineLearner, OnlineLearningConfig
from .streaming import StreamProcessor, StreamConfig, FeedbackSignal
from .feedback_processor import FeedbackProcessor, ProcessingStrategy
from .warmup import WarmupManager, WarmupConfig
from .evaluation import OnlineEvaluator, EvaluationMetrics, run_24hour_soak_test
from .interfaces import SafetyStatus, LearningPhase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bem2_online_learning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BEM2OnlineRunner:
    """
    Main runner for BEM 2.0 online learning system.
    
    Orchestrates all components according to TODO.md workflow:
    1. Warmup from AR1 checkpoint
    2. Initialize online learning components
    3. Process live feedback streams
    4. Run 24-hour soak test
    5. Monitor for +≥1% improvement and no canary regressions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.shutdown_requested = False
        
        # Initialize components
        self.online_learner: Optional[OnlineLearner] = None
        self.stream_processor: Optional[StreamProcessor] = None
        self.feedback_processor: Optional[FeedbackProcessor] = None
        self.evaluator: Optional[OnlineEvaluator] = None
        self.warmup_manager: Optional[WarmupManager] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize(self):
        """Initialize all online learning components"""
        logger.info("Initializing BEM 2.0 online learning system...")
        
        # Load configuration
        learning_config = OnlineLearningConfig(**self.config.get('online_learning', {}))
        stream_config = StreamConfig(**self.config.get('streaming', {}))
        warmup_config = WarmupConfig(**self.config.get('warmup', {}))
        
        # Initialize warmup manager
        self.warmup_manager = WarmupManager(warmup_config)
        
        # Initialize online learner
        self.online_learner = OnlineLearner(learning_config)
        
        # Initialize stream processor
        self.stream_processor = StreamProcessor(stream_config)
        
        # Initialize feedback processor
        processing_strategy = ProcessingStrategy(
            self.config.get('feedback_processing', {}).get('strategy', 'immediate')
        )
        self.feedback_processor = FeedbackProcessor(
            strategy=processing_strategy,
            batch_size=self.config.get('feedback_processing', {}).get('batch_size', 32)
        )
        
        # Initialize evaluator
        self.evaluator = OnlineEvaluator()
        
        logger.info("All components initialized successfully")
    
    async def warmup_from_checkpoint(self, checkpoint_path: str):
        """Perform warmup from AR1 checkpoint as specified in TODO.md"""
        logger.info(f"Starting warmup from AR1 checkpoint: {checkpoint_path}")
        
        # Load AR1 checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded AR1 checkpoint with keys: {list(checkpoint.keys())}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Run warmup process
        warmup_result = await self.warmup_manager.warmup_from_ar1(
            checkpoint=checkpoint,
            model=self.online_learner.model  # Assuming model is accessible
        )
        
        if warmup_result.success:
            logger.info(f"Warmup completed successfully:")
            logger.info(f"  - Fisher information computed: {warmup_result.fisher_computed}")
            logger.info(f"  - Canary baseline established: {warmup_result.canary_baseline_established}")
            logger.info(f"  - Safety validated: {warmup_result.safety_validated}")
            logger.info(f"  - Duration: {warmup_result.duration_seconds:.1f}s")
            
            # Set baseline metrics for evaluation
            if warmup_result.baseline_metrics:
                self.evaluator.set_baseline_metrics(warmup_result.baseline_metrics)
                logger.info("Baseline metrics established for soak test evaluation")
        else:
            logger.error(f"Warmup failed: {warmup_result.error_message}")
            raise RuntimeError(f"Warmup failed: {warmup_result.error_message}")
    
    async def start_streaming(self):
        """Start processing live feedback streams"""
        logger.info("Starting live feedback stream processing...")
        
        # Start the stream processor
        await self.stream_processor.start()
        
        # Main processing loop
        self.running = True
        
        while self.running and not self.shutdown_requested:
            try:
                # Process incoming feedback signals
                signals = await self.stream_processor.get_pending_signals(batch_size=32)
                
                if signals:
                    logger.debug(f"Processing {len(signals)} feedback signals")
                    
                    # Convert signals to training data
                    processed_feedback = await self.feedback_processor.process_signals(signals)
                    
                    if processed_feedback:
                        # Perform online learning step
                        for feedback in processed_feedback:
                            result = await self.online_learner.online_update_step(
                                batch=feedback.training_data,
                                feedback_score=feedback.aggregated_score
                            )
                            
                            # Update evaluation metrics
                            self.evaluator.update_metrics(result)
                            
                            # Log important updates
                            if result.update_applied:
                                logger.info(f"Online update applied: "
                                          f"Safety={result.safety_status.value}, "
                                          f"Phase={result.learning_phase.value}")
                            elif result.safety_status == SafetyStatus.CRITICAL:
                                logger.warning("Update rejected due to safety concerns")
                
                # Brief pause between processing cycles
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(1.0)  # Back off on errors
        
        logger.info("Streaming processing stopped")
    
    async def run_soak_test(self, duration_hours: float = 24.0):
        """Run the 24-hour soak test as specified in TODO.md"""
        logger.info(f"Starting {duration_hours}h soak test...")
        logger.info("Goal: +≥1% aggregate improvement with no canary regressions")
        
        if not self.evaluator.baseline_metrics:
            logger.error("Cannot run soak test without baseline metrics. Run warmup first.")
            return False
        
        # Start evaluation
        self.evaluator.start_evaluation()
        self.evaluator.target_soak_hours = duration_hours
        
        # Run soak test in parallel with streaming
        soak_task = asyncio.create_task(
            asyncio.to_thread(self.evaluator.run_soak_test)
        )
        stream_task = asyncio.create_task(self.start_streaming())
        
        # Monitor progress and handle completion
        try:
            # Wait for soak test completion or manual shutdown
            done, pending = await asyncio.wait(
                [soak_task, stream_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if soak_task in done:
                # Soak test completed
                soak_result = soak_task.result()
                logger.info("="*60)
                logger.info("SOAK TEST COMPLETED")
                logger.info("="*60)
                logger.info(soak_result.summary)
                
                if soak_result.success:
                    logger.info("✅ BEM 2.0 online learning system passed all requirements!")
                    logger.info(f"   Duration: {soak_result.duration_hours:.1f} hours")
                    logger.info(f"   Improvement: {soak_result.aggregate_improvement:+.2f}%")
                    logger.info(f"   Regressions: {'None' if not soak_result.has_regressions else 'Detected'}")
                else:
                    logger.warning("❌ Soak test failed to meet requirements")
                
                # Stop streaming
                self.running = False
                stream_task.cancel()
                
                return soak_result.success
            else:
                # Manual shutdown or error
                logger.info("Soak test interrupted")
                soak_task.cancel()
                return False
                
        except asyncio.CancelledError:
            logger.info("Soak test cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during soak test: {e}")
            return False
    
    async def generate_report(self):
        """Generate comprehensive evaluation report"""
        if not self.evaluator:
            logger.error("Evaluator not initialized")
            return
        
        report = self.evaluator.generate_evaluation_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"bem2_online_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {report_file}")
        
        # Log key metrics
        if 'baseline_comparison' in report:
            comparison = report['baseline_comparison']
            logger.info(f"Current improvement: {comparison['aggregate_improvement_percent']:+.2f}%")
            logger.info(f"Goal met: {comparison['improvement_goal_met']}")
            logger.info(f"Regressions: {comparison['has_regressions']}")
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down BEM 2.0 online learning system...")
        
        self.running = False
        
        # Stop streaming
        if self.stream_processor:
            await self.stream_processor.stop()
        
        # Save final checkpoint
        if self.online_learner:
            checkpoint_path = f"bem2_final_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            await self.online_learner.save_checkpoint(checkpoint_path)
            logger.info(f"Final checkpoint saved to: {checkpoint_path}")
        
        # Generate final report
        await self.generate_report()
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BEM 2.0 Online Learning Stream Runner")
    parser.add_argument("--config", type=str, default="config.json", 
                       help="Configuration file path")
    parser.add_argument("--warmup-from", type=str,
                       help="Path to AR1 checkpoint for warmup")
    parser.add_argument("--soak-test", action="store_true",
                       help="Run 24-hour soak test")
    parser.add_argument("--duration-hours", type=float, default=24.0,
                       help="Soak test duration in hours")
    parser.add_argument("--stream-only", action="store_true",
                       help="Run streaming processing only (no soak test)")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate evaluation report only")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    else:
        # Default configuration
        config = {
            "online_learning": {
                "learning_rate": 1e-5,
                "ewc_lambda": 1000.0,
                "replay_weight": 0.5,
                "gradient_clip": 1.0,
                "update_frequency": 10,
                "safety_threshold": 0.8
            },
            "streaming": {
                "max_queue_size": 10000,
                "batch_timeout": 5.0,
                "quality_threshold": 0.7
            },
            "warmup": {
                "fisher_samples": 1000,
                "canary_threshold": 0.9,
                "safety_margin": 0.1
            },
            "feedback_processing": {
                "strategy": "batched",
                "batch_size": 32,
                "aggregation_window": 300
            }
        }
        logger.warning(f"Configuration file not found, using defaults")
    
    # Initialize runner
    runner = BEM2OnlineRunner(config)
    
    try:
        # Initialize system
        await runner.initialize()
        
        # Warmup if checkpoint provided
        if args.warmup_from:
            if not Path(args.warmup_from).exists():
                logger.error(f"AR1 checkpoint not found: {args.warmup_from}")
                sys.exit(1)
            await runner.warmup_from_checkpoint(args.warmup_from)
        
        # Run based on mode
        if args.report_only:
            await runner.generate_report()
        elif args.soak_test:
            success = await runner.run_soak_test(args.duration_hours)
            sys.exit(0 if success else 1)
        elif args.stream_only:
            await runner.start_streaming()
        else:
            # Default: run streaming with periodic reporting
            logger.info("Starting continuous online learning (Ctrl+C to stop)")
            await runner.start_streaming()
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())