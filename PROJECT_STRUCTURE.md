# Project Structure Overview

This document provides a comprehensive overview of the Windows LLM Inference Benchmarking Suite project organization, suitable for recruiters and new contributors.

## 📁 Directory Layout

```
llm-windows-benchmark/
│
├── README.md                      # Main project overview
├── CHANGELOG.md                   # Version history and features
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
├── PROJECT_STRUCTURE.md          # This file
├── .gitignore                    # Git ignore rules
│
├── src/                          # Python source code (~1,850 LOC)
│   ├── __init__.py
│   ├── run_benchmark.py          # Entry point (CLI interface)
│   ├── main_orchestrator.py      # Master controller & lifecycle management
│   ├── task_handlers.py          # Task implementations (SQuAD, etc.)
│   ├── metrics_collector.py      # System monitoring & metrics
│   ├── runtimes.py               # Inference engine abstractions
│   ├── experiment_runner.py      # Experiment execution & coordination
│   ├── results_analyzer.py       # Results aggregation & reporting
│   └── config.json               # Experiment configuration
│
├── docs/                         # Documentation
│   ├── SETUP_GUIDE.md           # Installation & troubleshooting
│   ├── PROJECT_SUMMARY.md       # Technical deep-dive
│   ├── TASK_DOCUMENTATION.md    # Task implementation details
│   ├── COMPLETE_END_TO_END_GUIDE.md  # Step-by-step walkthrough
│   └── Ollama_Quantization.txt  # Ollama-specific setup
│
├── examples/                     # Example code and configurations
│   ├── config_example.json      # Example configuration
│   ├── quick_start.py           # Quick start Python example
│   └── (future: example notebooks, scripts)
│
├── scripts/                      # Utility scripts
│   └── install_llama_cpp_windows.ps1  # Windows setup helper
│
├── .github/                      # GitHub configuration
│   └── (future: workflow, templates)
│
├── requirements.txt              # Python dependencies
│
└── results/                      # Output directory (created at runtime)
    └── YYYYMMDD_HHMMSS/
        ├── experiment_results.jsonl    # Raw results
        ├── summary_report.txt          # Key findings
        ├── decision_matrix.txt         # Recommendations
        ├── results.csv                 # For Excel analysis
        └── benchmark_log.log           # Execution log
```

## 🔧 Core Modules Explained

### 1. **run_benchmark.py** - Entry Point (200 LOC)
**Responsibility**: CLI interface and benchmark orchestration

**Key Functions**:
- `main()` - Parse arguments and orchestrate execution
- `setup_logging()` - Configure logging levels
- `generate_configs()` - Create experiment configurations

**Usage**:
```bash
python src/run_benchmark.py                # Full benchmark
python src/run_benchmark.py --quick-test   # Quick validation
```

**Key Dependencies**: argparse, logging, config loading

---

### 2. **main_orchestrator.py** - Master Controller (200 LOC)
**Responsibility**: Environment setup, dataset loading, lifecycle management

**Key Classes**:
- `WindowsLLMBenchmark` - Main orchestrator class

**Key Methods**:
- `verify_windows_environment()` - Check Python, CUDA, dependencies
- `load_dataset()` - Load SQuAD v2.0 with caching
- `create_experiment_configs()` - Generate all experiment combinations

**System Checks**:
- Python version (3.10+)
- CUDA toolkit availability
- Required packages
- GPU detection

---

### 3. **task_handlers.py** - Task Implementations (300 LOC)
**Responsibility**: Define and execute benchmark tasks

**Key Classes**:
- `InferenceTask` (Abstract base)
- `SQuADTask` - Extractive QA evaluation
- `StructuredOutputTask` - JSON validation
- `LongContextTask` - Multi-passage reasoning

**Key Metrics**:
- F1 Score (token-level overlap)
- Exact Match (string matching)
- JSON Validity (structured output)

**Dataset**: SQuAD v2.0 (~250 examples per run)

---

### 4. **metrics_collector.py** - System Monitoring (250 LOC)
**Responsibility**: Real-time system metrics collection

**Key Classes**:
- `SystemMetricsCollector` - CPU, RAM, GPU monitoring
- `ColdStartMonitor` - Model loading latency
- `MetricsHistory` - Time-series storage

**Metrics Collected**:
- **Latency**: TTFT (ms), end-to-end latency
- **Throughput**: Tokens per second (TPS)
- **Memory**: Peak RAM, GPU memory, average usage
- **GPU**: Utilization, temperature
- **CPU**: Usage percentage

**Implementation**: Background threading for non-blocking collection

---

### 5. **runtimes.py** - Inference Engines (350 LOC)
**Responsibility**: Abstract inference interface and runtime implementations

**Key Classes**:
- `InferenceRuntime` (Abstract base)
- `LlamaCppRuntime` - llama.cpp via llama-cpp-python
- `OllamaRuntime` - HTTP API to Ollama service
- `PyTorchRuntime` - HuggingFace Transformers
- `ONNXRuntime` - ONNX Runtime (Windows-optimized)
- `RuntimeFactory` - Dynamic instantiation

**Key Methods** (all runtimes):
- `load_model(model_name, quantization)` - Model initialization
- `generate(prompt, max_tokens, temperature)` - Text generation
- `unload_model()` - Cleanup and memory release

**Quantization Support**:
- FP16 (full precision)
- INT8 (8-bit)
- Q4 (4-bit, llama.cpp only)

---

### 6. **experiment_runner.py** - Experiment Execution (300 LOC)
**Responsibility**: Execute individual experiments and manage batches

**Key Classes**:
- `ExperimentConfig` (dataclass) - Experiment configuration
- `ExperimentRunner` - Single experiment execution
- `ExperimentBatchRunner` - Batch coordination

**Workflow per Experiment**:
1. Initialize runtime + model
2. Warm GPU/CPU
3. Run task (20 prompts)
4. Collect metrics during execution
5. Evaluate results (F1, exact match)
6. Save to JSONL
7. Cleanup (unload model)

**Cold/Warm Start**:
- Warm start: Model already loaded
- Cold start: Fresh model load from disk

---

### 7. **results_analyzer.py** - Analysis & Reporting (250 LOC)
**Responsibility**: Aggregate results and generate reports

**Key Classes**:
- `ResultsAnalyzer` - Aggregation and analysis
- `DecisionMatrix` - Deployment recommendations

**Output Files**:
- `summary_report.txt` - Key statistics
- `decision_matrix.txt` - Recommendations by scenario
- `results.csv` - For Excel/external analysis
- `experiment_results.jsonl` - Raw data (1 line per experiment)

**Decision Scenarios**:
- Latency-critical chat
- Quality-first (high accuracy)
- Memory-constrained (limited RAM)
- Structured output (JSON validation)
- Enterprise deployment

---

## 📊 Data Flow

```
run_benchmark.py
    │
    ├──> Load config.json
    │
    ├──> main_orchestrator.py
    │    ├─ Verify environment
    │    ├─ Load SQuAD dataset (cached)
    │    └─ Generate experiment configs (192 total)
    │
    ├──> For Each Experiment:
    │    │
    │    ├──> experiment_runner.py
    │    │    ├─ Initialize runtime (runtimes.py)
    │    │    ├─ Load model with quantization
    │    │    │
    │    │    ├─ For Each Trial (3 times):
    │    │    │  ├─ metrics_collector.py (start)
    │    │    │  ├─ task_handlers.py (generate prompts)
    │    │    │  ├─ runtimes.py (inference)
    │    │    │  ├─ metrics_collector.py (collect TTFT, TPS, memory)
    │    │    │  ├─ task_handlers.py (evaluate F1, exact match)
    │    │    │  └─ Save to results.jsonl
    │    │    │
    │    │    └─ Cleanup (unload model)
    │    │
    │    └──> Continue to next experiment
    │
    └──> results_analyzer.py
         ├─ Aggregate metrics
         ├─ Generate summary_report.txt
         ├─ Create decision_matrix.txt
         └─ Export results.csv
```

## 🔄 Configuration System

**File**: `src/config.json`

**Key Sections**:
```json
{
  "models": [...],              // 5 models: Qwen, Llama, Phi, Mistral
  "runtimes": [...],            // 4 runtimes: llama.cpp, Ollama, PyTorch, ONNX
  "quantizations": [...],       // 3 types: FP16, INT8, Q4
  "workload_types": [...],      // 3 tasks: chat, structured, long-context
  "experiments": {...},         // Trial count, timeout, max tokens
  "dataset": {...}              // SQuAD configuration
}
```

**Total Combinations**: 5 × 4 × 3 × 3 = 180 base configs + cold-start variants ≈ 192

## 📚 Dependencies

**Core ML**:
- torch >= 2.0.0
- transformers >= 4.35.0
- datasets >= 2.14.0

**Inference Runtimes**:
- llama-cpp-python (llama.cpp)
- requests (Ollama HTTP)
- onnxruntime-gpu (ONNX)

**System Monitoring**:
- psutil

**Development**:
- jupyter, matplotlib, scikit-learn (for analysis)

**Optional**:
- bitsandbytes (quantization)
- auto-gptq (GPTQ quantization)

See `requirements.txt` for full list.

## 📊 Metrics & Evaluation

### System-Level Metrics
- **TTFT (ms)**: Time to First Token
- **TPS**: Tokens per second (throughput)
- **Memory (MB)**: Peak RAM and GPU usage
- **Cold Start (ms)**: Model load + first inference

### Task-Level Metrics
- **F1 Score**: Token-level overlap (0-1)
- **Exact Match**: Percentage of perfect matches (0-100)
- **JSON Validity**: Percentage of valid structured output

## 🎯 Key Classes & Patterns

### Abstract Base Classes
```python
class InferenceRuntime(ABC):
    @abstractmethod
    def load_model(self, model_name, quantization) -> bool:
        pass
    
    @abstractmethod
    def generate(self, prompt, max_tokens, **kwargs) -> tuple[str, int]:
        pass

class InferenceTask(ABC):
    @abstractmethod
    def create_prompt(self, data) -> str:
        pass
    
    @abstractmethod
    def evaluate(self, prediction, reference) -> dict:
        pass
```

### Factory Pattern
```python
runtime = RuntimeFactory.create(
    runtime_name="pytorch",
    device="cuda",
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
)
```

### Configuration Pattern
```python
@dataclass
class ExperimentConfig:
    model_name: str
    runtime: str
    quantization: str
    workload_type: str
    trial_number: int
    is_cold_start: bool
```

## 🔍 Testing Strategy

### Quick Test (`--quick-test`)
- Validates environment
- Tests single model with 1 example
- Runs in < 5 minutes

### Unit Tests (future)
- Test individual modules
- Mock external dependencies
- Validate metrics calculations

### Integration Tests (future)
- End-to-end benchmark runs
- Results validation
- Report generation

## 📈 Performance Characteristics

- **Total Experiments**: ~192 configurations
- **Trials per Experiment**: 3 (1 cold-start, 2 warm-start)
- **Total Runs**: ~576 inference sessions
- **Estimated Time**: 8-12 hours (RTX 4060 baseline)
- **Output Size**: 10-50 MB (JSONL + reports)
- **Code Size**: ~1,850 lines of Python

## 🚀 Getting Started for Contributors

1. **Understand the Architecture**
   - Read this document
   - Review docstrings in source files
   - Check examples in `examples/`

2. **Run a Test**
   ```bash
   python src/run_benchmark.py --quick-test
   ```

3. **Explore the Code**
   - Start with `run_benchmark.py` (entry point)
   - Follow the data flow through modules
   - Review docstrings and type hints

4. **Make Contributions**
   - Follow guidelines in CONTRIBUTING.md
   - Add tests for new features
   - Update documentation

## 📖 Further Reading

- **[README.md](README.md)** - Project overview and quick start
- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Installation instructions
- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - Technical deep-dive
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines
- **[docs/COMPLETE_END_TO_END_GUIDE.md](docs/COMPLETE_END_TO_END_GUIDE.md)** - Full walkthrough

---

**Last Updated**: May 5, 2025
**Version**: 1.0.0
