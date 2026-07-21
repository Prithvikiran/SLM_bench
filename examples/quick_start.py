"""
Quick Start Example: Running the LLM Benchmarking Suite

This example demonstrates how to run a simple benchmark configuration.
For full documentation, see docs/SETUP_GUIDE.md
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main_orchestrator import WindowsLLMBenchmark
from experiment_runner import ExperimentConfig, ExperimentRunner


def main():
    """
    Run a simple benchmark with a single model and configuration.
    """

    print("=" * 60)
    print("Windows LLM Benchmarking Suite - Quick Start")
    print("=" * 60)

    # Configuration for a simple experiment
    config = ExperimentConfig(
        model_name="Qwen2.5-1.5B-Instruct",
        runtime="pytorch",
        quantization="fp16",
        workload_type="interactive_chat",
        trial_number=1,
        is_cold_start=False
    )

    print(f"\nBenchmarking Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Runtime: {config.runtime}")
    print(f"  Quantization: {config.quantization}")
    print(f"  Workload: {config.workload_type}")

    # Initialize benchmark orchestrator
    print("\nInitializing benchmark environment...")
    benchmark = WindowsLLMBenchmark()

    # Verify setup
    if not benchmark.verify_windows_environment():
        print("❌ Windows environment verification failed!")
        return False

    print("✓ Windows environment verified")

    # Load dataset
    print("\nLoading SQuAD v2.0 dataset...")
    if not benchmark.load_dataset(num_samples=50):
        print("❌ Dataset loading failed!")
        return False

    print(f"✓ Loaded {len(benchmark.dataset)} examples")

    # Run experiment
    print(f"\nRunning benchmark experiment...")
    runner = ExperimentRunner(config, benchmark.dataset)

    try:
        results = runner.run()

        if results["status"] == "completed":
            print("✓ Experiment completed successfully!")

            # Display key metrics
            metrics = results.get("metrics", {})
            print("\nKey Results:")
            print(f"  F1 Score: {metrics.get('task_metrics', {}).get('f1', 'N/A'):.3f}")
            print(f"  TTFT (ms): {metrics.get('latency_stats_ms', {}).get('mean', 'N/A'):.1f}")
            print(f"  Throughput (tok/s): {metrics.get('throughput_stats_tps', {}).get('mean', 'N/A'):.1f}")
            print(f"  Memory (MB): {metrics.get('memory_stats_mb', {}).get('mean', 'N/A'):.0f}")

            return True
        else:
            print(f"❌ Experiment failed: {results.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Exception during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    print("Quick Start Complete" if success else "Quick Start Failed")
    print("=" * 60)

    sys.exit(0 if success else 1)
