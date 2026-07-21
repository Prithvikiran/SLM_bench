"""
Integration Script - Main Entry Point
Coordinates all benchmarking components
"""

import logging
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Import all modules
from main_orchestrator import BenchmarkOrchestrator
from task_handlers import SQuADTask
from metrics_collector import SystemMetricsCollector
from runtimes import RuntimeFactory
from experiment_runner import ExperimentConfig, ExperimentBatchRunner
from results_analyzer import generate_all_reports


def setup_logging(log_dir: Path):
    """Configure logging"""
    log_file = log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def _model_id_for_runtime(model_info: dict, runtime: str) -> str:
    """Resolve the model identifier each runtime expects."""
    if runtime == "ollama":
        return model_info.get("ollama_model") or model_info["name"]
    # PyTorch / ONNX / llama.cpp use HuggingFace repo id or local path.
    return model_info.get("hf_id", model_info["name"])


def generate_experiment_configs(config: dict) -> list:
    """Generate all experiment configurations"""
    configs = []
    
    all_models = []
    for tier_models in config['models'].values():
        all_models.extend(tier_models)
    
    trial_num = 0
    
    for model_info in all_models:
        for runtime in config['runtimes']:
            model_name = _model_id_for_runtime(model_info, runtime)
            for quantization in config['quantizations']:
                for workload in config['workloads']:
                    # Warm start trials
                    for trial in range(config['experiments']['trials_per_config']):
                        trial_num += 1
                        configs.append(ExperimentConfig(
                            model_name=model_name,
                            runtime=runtime,
                            quantization=quantization,
                            workload_type=workload,
                            trial_number=trial + 1,
                            is_cold_start=False
                        ))
                    
                    # Cold start trial (once per config)
                    if config['experiments']['cold_start']:
                        trial_num += 1
                        configs.append(ExperimentConfig(
                            model_name=model_name,
                            runtime=runtime,
                            quantization=quantization,
                            workload_type=workload,
                            trial_number=1,
                            is_cold_start=True
                        ))
    
    return configs


def _onnx_path_for_config(model_name: str, quantization: str) -> Path:
    """Path convention used by ONNXRuntime._get_onnx_path()."""
    return Path(f"./models/{model_name}-{quantization}.onnx")


def _try_export_missing_fp16_onnx(model_name: str, target_path: Path, logger: logging.Logger) -> bool:
    """Best-effort export of a missing fp16 ONNX model via Optimum.

    Returns True if `target_path` is created and exists.
    """
    if target_path.exists():
        return True

    try:
        from optimum.exporters.onnx import main_export
    except Exception as e:
        logger.warning(
            f"Cannot auto-export ONNX for {model_name}: optimum ONNX exporter unavailable ({e})"
        )
        return False

    export_dir = target_path.parent / f"{Path(model_name).name}_onnx_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Auto-exporting ONNX fp16 for {model_name} ... this can take a while.")

    # Prefer CausalLM export with kv-cache graph when supported.
    attempts = [
        ("text-generation-with-past", False),
        ("text-generation", False),
        ("text-generation-with-past", True),
        ("text-generation", True),
    ]
    exported = False
    last_error = None
    for task, no_post_process in attempts:
        try:
            if no_post_process:
                logger.info(
                    f"Retrying ONNX export for {model_name} with task='{task}' and no_post_process=True."
                )
            main_export(
                model_name_or_path=model_name,
                output=export_dir,
                task=task,
                no_post_process=no_post_process,
            )
            exported = True
            break
        except Exception as e:
            last_error = e
            if no_post_process:
                logger.warning(
                    f"ONNX export failed for {model_name} with task='{task}' and no_post_process=True: {e}"
                )
            else:
                logger.warning(
                    f"ONNX export failed for {model_name} with task='{task}': {e}"
                )

    if not exported:
        logger.warning(f"ONNX export failed for {model_name}: {last_error}")
        return False

    # Try common filenames produced by Optimum.
    candidates = [
        export_dir / "model.onnx",
        export_dir / "decoder_model_merged.onnx",
        export_dir / "decoder_model.onnx",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        all_onnx = list(export_dir.rglob("*.onnx"))
        source = all_onnx[0] if all_onnx else None

    if source is None:
        logger.warning(f"ONNX export produced no .onnx file for {model_name}")
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    logger.info(f"Created ONNX artifact: {target_path}")
    return target_path.exists()


def _filter_missing_onnx_configs(configs: list, logger: logging.Logger) -> list:
    """Skip ONNX experiments whose model artifact does not exist locally.

    For missing fp16 artifacts, attempt a one-time auto-export via Optimum.
    """
    kept = []
    missing = []
    attempted_exports = set()
    for cfg in configs:
        if cfg.runtime != "onnx_runtime":
            kept.append(cfg)
            continue
        onnx_path = _onnx_path_for_config(cfg.model_name, cfg.quantization)
        if onnx_path.exists():
            kept.append(cfg)
            continue

        # Auto-export only for fp16; q4/fp8 need dedicated quantized ONNX flows.
        export_key = (cfg.model_name, cfg.quantization)
        if cfg.quantization == "fp16" and export_key not in attempted_exports:
            attempted_exports.add(export_key)
            if _try_export_missing_fp16_onnx(cfg.model_name, onnx_path, logger):
                kept.append(cfg)
                continue
        elif cfg.quantization == "fp16" and export_key in attempted_exports and onnx_path.exists():
            kept.append(cfg)
        else:
            missing.append((cfg.model_name, cfg.quantization, str(onnx_path)))

    if missing:
        logger.warning(
            f"Skipping {len(missing)} ONNX experiment configs because model files are missing."
        )
        seen = set()
        for model_name, quant, path in missing:
            key = (model_name, quant, path)
            if key in seen:
                continue
            seen.add(key)
            logger.warning(
                f"  Missing ONNX file for {model_name} [{quant}]: {path}"
            )
    return kept


def run_full_benchmark():
    """Run complete benchmarking suite"""
    logger = setup_logging(Path("."))
    
    logger.info("=" * 80)
    logger.info("WINDOWS LLM INFERENCE BENCHMARKING SUITE")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config_file = Path("config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {config_file}")
        
        # Create orchestrator
        orchestrator = BenchmarkOrchestrator(str(config_file))
        
        # Setup and load dataset
        logger.info("\nLoading SQuAD v2.0 dataset...")
        from datasets import load_dataset
        dataset = load_dataset('squad_v2')
        
        # Sample dataset
        squad_data = {
            'contexts': [],
            'questions': [],
            'answers': [],
            'ids': []
        }
        
        num_samples = config['squad_samples']
        for idx, item in enumerate(dataset['validation']):
            if idx >= num_samples:
                break
            squad_data['contexts'].append(item['context'])
            squad_data['questions'].append(item['question'])
            squad_data['answers'].append(item['answers'])
            squad_data['ids'].append(item['id'])
        
        logger.info(f"Loaded {len(squad_data['contexts'])} SQuAD v2 examples")
        
        # Generate experiment configs
        logger.info("\nGenerating experiment configurations...")
        configs = generate_experiment_configs(config)
        configs = _filter_missing_onnx_configs(configs, logger)
        if not configs:
            logger.error(
                "No runnable experiment configs after ONNX file checks. "
                "Place ONNX files under ./models/ or adjust runtimes/quantizations."
            )
            return False
        logger.info(f"Generated {len(configs)} experiment configurations")
        logger.info(f"Estimated runtime: {len(configs) * 2 / 60:.1f} hours (2 min per config avg)")
        
        # Run experiments
        logger.info("\nStarting benchmark runs...")
        batch_runner = ExperimentBatchRunner(orchestrator.results_dir)
        results = batch_runner.run_batch(configs, squad_data)
        
        # Generate reports
        logger.info("\nGenerating analysis reports...")
        generate_all_reports(orchestrator.results_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Results saved to: {orchestrator.results_dir}")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
        return False


def run_quick_test():
    """Run quick validation test"""
    logger = setup_logging(Path("."))
    
    logger.info("=" * 80)
    logger.info("QUICK VALIDATION TEST")
    logger.info("=" * 80)
    
    try:
        # Test imports
        logger.info("\n1. Testing imports...")
        from main_orchestrator import BenchmarkOrchestrator
        from task_handlers import SQuADTask
        from metrics_collector import SystemMetricsCollector
        from runtimes import RuntimeFactory
        logger.info("OK: All imports successful")
        
        # Test system info
        logger.info("\n2. Checking system configuration...")
        import platform
        import torch
        logger.info(f"  OS: {platform.system()} {platform.release()}")
        logger.info(f"  Python: {platform.python_version()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("  GPU: Not detected (CPU mode)")
        
        # Test task handler
        logger.info("\n3. Testing SQuAD task handler...")
        task = SQuADTask()
        test_context = "The capital of France is Paris."
        test_question = "What is the capital of France?"
        prompt = task.create_prompt(test_context, test_question)
        logger.info("OK: Task handler")
        
        # Test metrics collector
        logger.info("\n4. Testing metrics collector...")
        metrics = SystemMetricsCollector()
        metrics.start_monitoring()
        import time
        time.sleep(1)
        metrics.stop_monitoring()
        logger.info("OK: Metrics collector")
        
        # Test runtime factory (uses config.json runtimes + first model)
        logger.info("\n5. Testing runtime factory...")
        with open(Path("config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        all_models = []
        for tier_models in cfg["models"].values():
            all_models.extend(tier_models)
        if not all_models:
            logger.warning("No models in config.json — skipping runtime factory check")
        else:
            first = all_models[0]
            quant = (cfg.get("quantizations") or ["fp16"])[0]
            for runtime in cfg.get("runtimes") or []:
                model_id = _model_id_for_runtime(first, runtime)
                rt = RuntimeFactory.create(runtime, model_id, quant)
                if rt:
                    logger.info(f"OK: {runtime} (model id: {model_id})")
                else:
                    logger.warning(f"Missing: {runtime}")
        
        logger.info("\n" + "=" * 80)
        logger.info("QUICK TEST PASSED - Ready for full benchmark")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Quick test failed: {str(e)}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Windows LLM Inference Benchmarking Suite'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick validation test'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='Configuration file (default: config.json)'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        success = run_quick_test()
    else:
        success = run_full_benchmark()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
