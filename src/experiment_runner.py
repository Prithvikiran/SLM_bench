"""
Experiment Runner
Executes a single benchmarking experiment configuration
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from metrics_collector import SystemMetricsCollector, ColdStartMonitor
from task_handlers import get_task_handler, SQuADTask
from runtimes import RuntimeFactory

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    model_name: str
    runtime: str
    quantization: str
    workload_type: str
    trial_number: int
    is_cold_start: bool = False


class ExperimentRunner:
    """Runs a single benchmarking experiment"""
    
    def __init__(self, config: ExperimentConfig, squad_data: Dict, output_dir: Path):
        self.config = config
        self.squad_data = squad_data
        self.output_dir = output_dir
        self.runtime = None
        self.task = None
        self.metrics = SystemMetricsCollector()
        self.cold_start_monitor = ColdStartMonitor() if config.is_cold_start else None
        
    def run(self) -> Dict[str, Any]:
        """Execute the experiment"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {self.config.model_name} | {self.config.runtime} | "
                   f"{self.config.quantization} | {self.config.workload_type} | Trial {self.config.trial_number}")
        logger.info(f"{'='*60}")
        
        result = {
            'config': asdict(self.config),
            'status': 'pending',
            'metrics': {},
            'errors': []
        }
        
        try:
            # Setup phase
            if not self._setup():
                result['status'] = 'setup_failed'
                return result
            
            # Run experiment
            if self.config.is_cold_start:
                experiment_result = self._run_cold_start_experiment()
            else:
                experiment_result = self._run_warm_start_experiment()
            
            result['metrics'] = experiment_result
            result['status'] = 'completed'
            
            # Cleanup
            self._cleanup()
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            result['status'] = 'failed'
            result['errors'].append(str(e))
            self._cleanup()
        
        return result
    
    def _setup(self) -> bool:
        """Setup experiment (load model, prepare data)"""
        try:
            logger.info(f"Setting up runtime: {self.config.runtime}")
            
            # Create runtime
            self.runtime = RuntimeFactory.create(
                self.config.runtime,
                self.config.model_name,
                self.config.quantization
            )
            
            if not self.runtime:
                logger.error("Failed to create runtime")
                return False
            
            # Load model
            if self.config.is_cold_start:
                self.cold_start_monitor.start_load()
            
            if not self.runtime.load_model():
                logger.error("Failed to load model")
                return False
            
            if self.config.is_cold_start:
                self.cold_start_monitor.end_load()
            
            # Setup task
            self.task = get_task_handler(self.config.workload_type)
            
            logger.info("Setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def _run_warm_start_experiment(self) -> Dict:
        """Run warm-start experiment (model already loaded)"""
        predictions = []
        all_metrics = []
        
        # Sample 20 examples for benchmarking (not all 250)
        sample_size = min(20, len(self.squad_data['contexts']))
        
        for idx in range(sample_size):
            context = self.squad_data['contexts'][idx]
            question = self.squad_data['questions'][idx]
            reference_answers = self.squad_data['answers'][idx]
            
            # Create prompt based on workload type
            if self.config.workload_type == "interactive_chat":
                prompt = self.task.create_prompt(context, question)
            elif self.config.workload_type == "structured_output":
                prompt = self.task.create_prompt(context, question)
            else:  # long_context
                # For long context, use current + next 2 passages
                passages = self.squad_data['contexts'][idx:min(idx+3, len(self.squad_data['contexts']))]
                prompt = self.task.create_prompt(passages, question)
            
            # Run inference
            self.metrics.reset()
            self.metrics.start_monitoring()
            self.metrics.start_request_timer()
            
            try:
                generated_text, tokens = self.runtime.generate(
                    prompt,
                    max_tokens=64,
                    temperature=0.1,
                    top_p=0.9
                )
                
                self.metrics.record_first_token()
                self.metrics.output_tokens = tokens
                self.metrics.end_request_timer()
                self.metrics.stop_monitoring()
                
                # Extract answer
                prediction = self.task.extract_answer(generated_text)
                predictions.append(prediction)
                
                # Get metrics
                trial_metrics = self.metrics.get_all_metrics()
                trial_metrics['sample_id'] = self.squad_data['ids'][idx]
                all_metrics.append(trial_metrics)
                
                logger.info(f"  Sample {idx+1}/{sample_size}: TTFT={trial_metrics['latency'].get('ttft_ms', 0):.1f}ms, "
                           f"TPS={trial_metrics['throughput'].get('tokens_per_second', 0):.1f}")
            
            except Exception as e:
                logger.warning(f"  Sample {idx+1} failed: {e}")
                all_metrics.append({
                    'sample_id': self.squad_data['ids'][idx],
                    'error': str(e)
                })
        
        # Evaluate task performance
        references = self.squad_data['answers'][:sample_size]
        task_metrics = self.task.batch_evaluate(predictions, references)
        
        # Aggregate metrics
        return self._aggregate_metrics(all_metrics, task_metrics)
    
    def _run_cold_start_experiment(self) -> Dict:
        """Run cold-start experiment (time model load + first inference)"""
        if not self.cold_start_monitor:
            logger.error("Cold start monitor not initialized")
            return {}
        
        try:
            # First inference (model already loaded in _setup)
            self.cold_start_monitor.start_inference()
            
            context = self.squad_data['contexts'][0]
            question = self.squad_data['questions'][0]
            prompt = self.task.create_prompt(context, question)
            
            generated_text, tokens = self.runtime.generate(prompt)
            
            self.cold_start_monitor.end_inference()
            
            return {
                'cold_start_metrics': self.cold_start_monitor.get_metrics()
            }
        
        except Exception as e:
            logger.error(f"Cold start experiment failed: {e}")
            return {'error': str(e)}
    
    def _aggregate_metrics(self, all_metrics: List[Dict], task_metrics: Dict) -> Dict:
        """Aggregate metrics across all samples"""
        latencies = []
        throughputs = []
        memory_values = []
        
        for trial in all_metrics:
            if 'error' in trial:
                continue
            
            latencies.append(trial['latency'].get('ttft_ms', 0))
            throughputs.append(trial['throughput'].get('tokens_per_second', 0))
            memory_values.append(trial['memory'].get('peak_memory_mb', 0))
        
        if not latencies:
            return {
                'task_metrics': task_metrics,
                'error': 'No successful samples'
            }
        
        # Calculate statistics
        def calc_stats(values):
            return {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': sorted(values)[len(values)//2]
            }
        
        return {
            'task_metrics': task_metrics,
            'latency_stats_ms': calc_stats(latencies),
            'throughput_stats_tps': calc_stats(throughputs),
            'memory_stats_mb': calc_stats(memory_values),
            'samples_completed': len(latencies),
            'all_trial_metrics': all_metrics
        }
    
    def _cleanup(self):
        """Cleanup after experiment"""
        try:
            if self.metrics.monitoring:
                self.metrics.stop_monitoring()
            
            if self.runtime:
                logger.info(f"Unloading {self.config.model_name}")
                self.runtime.unload_model()
        
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class ExperimentBatchRunner:
    """Runs a batch of experiments"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = []
    
    def run_batch(self, configs: List[ExperimentConfig], squad_data: Dict) -> List[Dict]:
        """Run multiple experiments"""
        logger.info(f"\nRunning {len(configs)} experiments...")
        
        for i, config in enumerate(configs):
            logger.info(f"\n[{i+1}/{len(configs)}] Starting experiment...")
            
            runner = ExperimentRunner(config, squad_data, self.output_dir)
            result = runner.run()
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
            
            # Brief pause between experiments
            time.sleep(2)
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / "experiment_results.jsonl"
        
        with open(output_file, 'a') as f:
            for result in self.results[-1:]:  # Save last result
                f.write(json.dumps(result) + '\n')
