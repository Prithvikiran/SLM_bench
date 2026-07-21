# Windows LLM Inference Benchmarking Suite

A production-ready Python benchmarking framework for evaluating local large language model (LLM) inference on Windows hardware. This project systematically evaluates multiple runtimes, quantization methods, and model sizes using the SQuAD v2.0 extractive question-answering task.

**Publication**: [Benchmarking Local Small Language Model Inference on Windows: Runtime, Quantization, and Deployment Trade-offs](docs/PROJECT_SUMMARY.md) — CSCI 5922, University of Colorado Boulder, 2025

---

## 🎯 Key Innovation

**First systematic Windows-focused benchmark** for local LLM inference, combining:
- **4 inference runtimes** (llama.cpp, Ollama, PyTorch, ONNX Runtime)
- **4 distinct LLM architectures** (Qwen, Llama, Mistral, Phi)
- **Reproducible evaluation** on SQuAD v2.0 extractive QA task
- **Windows-specific metrics** (cold-start latency, DLL loading overhead)
- **Production decision framework** for IT practitioners and ML engineers

### Why It Matters
- Windows represents 75% of desktop OS market, yet lacks comprehensive local LLM benchmarks
- No existing work covers Windows + practical inference runtimes + reproducible task evaluation
- Direct applicability to enterprise deployments, edge computing, and cost optimization

---

## 📊 Quick Results

Baseline system: **Windows 11, RTX 4060, Intel i7-13700K, 32GB RAM**

| Scenario | Recommended Config | TTFT | TPS | F1 Score | Memory |
|----------|-------------------|------|-----|----------|--------|
| **Latency-Critical Chat** | llama.cpp + Qwen-1.5B (Q4) | 22ms | 155 tok/s | 0.81 | 650MB |
| **Quality-First** | PyTorch + Qwen-7B (FP16) | 105ms | 48 tok/s | 0.84 | 14.2GB |
| **Memory-Constrained** | Phi-3.5-mini (ONNX, INT8) | 35ms | 118 tok/s | 0.79 | 2.6GB |
| **Structured Output** | ONNX + Phi-3.5-mini (JSON) | 30ms | 120 tok/s | 99% valid | 2.5GB |

---

## 🏗️ Architecture

```
src/
├── run_benchmark.py              [Entry Point]
│   ├─ CLI interface with --quick-test mode
│   └─ Experiment orchestration
│
├── main_orchestrator.py          [Master Controller]
│   ├─ Windows environment verification
│   ├─ Dataset loading & caching
│   └─ Experiment lifecycle management
│
├── task_handlers.py              [Task Implementations]
│   ├─ SQuADTask: Extractive QA with F1/Exact Match
│   ├─ StructuredOutputTask: JSON validation
│   └─ LongContextTask: Multi-passage reasoning
│
├── metrics_collector.py          [System Monitoring]
│   ├─ Real-time CPU/GPU/memory tracking
│   ├─ Time-to-First-Token (TTFT) measurement
│   └─ Background thread monitoring
│
├── runtimes.py                   [Inference Engines]
│   ├─ LlamaCppRuntime: GGUF models via llama-cpp-python
│   ├─ OllamaRuntime: HTTP API to Ollama service
│   ├─ PyTorchRuntime: HuggingFace Transformers
│   ├─ ONNXRuntime: ONNX models (Windows-optimized)
│   └─ RuntimeFactory: Dynamic instantiation
│
├── experiment_runner.py          [Experiment Execution]
│   ├─ Single experiment orchestration
│   ├─ Trial management & averaging
│   └─ Warm/cold-start experiment logic
│
├── results_analyzer.py           [Analysis & Reporting]
│   ├─ Results aggregation across experiments
│   ├─ Summary report generation
│   └─ Decision matrix creation
│
└── config.json                   [Configuration]
    ├─ Models, runtimes, quantizations
    └─ Workload parameters
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
# Clone repository
git clone https://github.com/yourusername/llm-windows-benchmark.git
cd llm-windows-benchmark

# Create virtual environment
python -m venv benchmark_env
benchmark_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Validate Setup (5 minutes)
```bash
python src/run_benchmark.py --quick-test
```
Verifies Python version, CUDA toolkit, all dependencies, and GPU availability.

### 3. Run Full Benchmark (8-12 hours)
```bash
python src/run_benchmark.py
```
Results saved to `results/YYYYMMDD_HHMMSS/`

### 4. Analyze Results
```bash
# View summary report
cat results/YYYYMMDD_HHMMSS/summary_report.txt

# View deployment recommendations
cat results/YYYYMMDD_HHMMSS/decision_matrix.txt

# Open CSV in Excel for custom analysis
results/YYYYMMDD_HHMMSS/results.csv
```

---

## 📋 Models Evaluated

### Small Tier (1-3.8B parameters)
- **Qwen2.5-1.5B-Instruct** — Strong SQuAD performance, excellent quantization
- **Llama-3.2-1B-Instruct** — Meta's recent efficient model
- **Phi-3.5-mini-Instruct** — Microsoft-optimized, ONNX focus (novel contribution)

### Medium Tier (7B parameters)
- **Qwen2.5-7B-Instruct** — SOTA for size, same family as 1.5B
- **Mistral-7B-Instruct-v0.3** — Different attention mechanism, widely benchmarked

---

## ⚙️ Inference Runtimes

| Runtime | Type | Best For | Windows | GPU Support |
|---------|------|----------|---------|-------------|
| **llama.cpp** | Lightweight C++ | Cold-start, CPU inference | ✓ Excellent | ✓ Partial |
| **Ollama** | Developer API | Quick prototyping | ✓ Great | ✓ Good |
| **PyTorch** | Full framework | Flexibility, research | ✓ Good | ✓ Excellent |
| **ONNX Runtime** | Optimized graphs | Enterprise, Windows-native | ✓ Best-in-class | ✓ Good |

---

## 🔬 Evaluation Metrics

### System-Level Metrics
- **Time-to-First-Token (TTFT)** — Latency before first output token
- **Throughput (TPS)** — Tokens generated per second
- **Memory Usage** — Peak RAM and GPU memory consumption
- **Cold Start** — Model load time + first inference latency

### Task-Level Metrics (SQuAD v2.0)
- **F1 Score** — Token-level overlap with reference answers
- **Exact Match** — Normalized answer string matching
- **JSON Validity** — Percentage of valid structured output

---

## 📦 Quantization Methods

- **FP16** — Full precision (baseline for quality comparison)
- **INT8** — 8-bit quantization (moderate compression)
- **Q4** — 4-bit quantization (maximum compression, llama.cpp)

---

## 💾 Installation Requirements

### System Requirements
- **OS**: Windows 10 or later (tested on Windows 11)
- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB recommended)
- **GPU** (optional): NVIDIA CUDA 11.8+ (for GPU acceleration)

### Dependencies
```bash
# Core ML frameworks
pip install torch>=2.0.0 transformers>=4.35.0 datasets>=2.14.0

# Inference runtimes
pip install llama-cpp-python onnxruntime-gpu requests

# System monitoring
pip install psutil

# See requirements.txt for full list
```

Detailed setup instructions: [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)

---

## 🔄 Workflow Example

```
1. Run Configuration
   │
   ├─ Load 4 models
   ├─ Load 4 runtimes
   ├─ Load 3 quantizations
   ├─ Load 3 workload types
   └─ Total: ~192 experiment configurations

2. For Each Experiment
   │
   ├─ Initialize runtime + model
   ├─ Collect baseline metrics
   ├─ Generate 20 prompts from SQuAD
   ├─ Run inference, measure TTFT, throughput
   ├─ Evaluate F1 / Exact Match
   ├─ Save results as JSONL
   └─ Cleanup (unload model, free GPU)

3. Post-Processing
   │
   ├─ Aggregate metrics across trials
   ├─ Generate summary report
   ├─ Create decision matrix
   └─ Export to CSV
```

---

## 📊 Output Format

Results directory structure:
```
results/
└── 20250505_143022/
    ├── experiment_results.jsonl    # Raw data (1 line per experiment)
    ├── summary_report.txt          # Key statistics & insights
    ├── decision_matrix.txt         # Deployment recommendations
    ├── results.csv                 # For Excel/external analysis
    ├── squad_dataset_cache.json    # Cached dataset
    └── benchmark_log.log           # Detailed execution log
```

---

## 🛠️ Windows-Specific Features

1. **Cold/Warm Start Analysis** — DLL loading overhead measured separately
2. **CPU-Only Fallback** — Support for integrated GPU + discrete GPU
3. **Power State Monitoring** — Thermal throttling detection
4. **Task Scheduler Integration** — Guidance for batch scheduling
5. **ONNX Optimization** — Microsoft-na