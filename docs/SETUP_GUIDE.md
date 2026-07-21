# Windows LLM Inference Benchmarking Suite - Setup Guide

## Quick Start

### 1. System Requirements
- **OS**: Windows 10/11
- **CPU**: Intel i5/i7 or AMD equivalent (8+ cores)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: Optional but recommended (NVIDIA RTX 4060 or better)
- **Storage**: 100GB (for models)
- **Python**: 3.10+

### 2. Installation

#### Step 1: Clone repository and setup environment
```bash
# Create virtual environment
python -m venv benchmark_env
benchmark_env\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft bitsandbytes

# Install runtime-specific dependencies
pip install llama-cpp-python
pip install requests  # For Ollama API
pip install onnxruntime-gpu  # Use onnxruntime for CPU-only
pip install optimum[exporters-torch]  # For ONNX export
```

#### Step 2: Install inference engines

**llama.cpp**
```bash
# Already installed via llama-cpp-python
# Verify: python -c "from llama_cpp import Llama; print('OK')"
```

**Ollama**
```bash
# Download from https://ollama.com
# Run installer and start Ollama service
ollama serve
# In new terminal: ollama pull <model_name>
```

**ONNX Runtime**
```bash
# Already installed above
pip install onnxruntime-gpu  # GPU support
# or
pip install onnxruntime  # CPU only
```

#### Step 3: Prepare models
```bash
# Create models directory
mkdir models

# Download models (will cache automatically in huggingface cache)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')"

# For GGUF models (llama.cpp), convert models:
pip install llama-cpp-python
# Conversion script will follow
```

### 3. Windows Optimization (Recommended)

Before running benchmarks, optimize Windows for consistent results:

```powershell
# Run as Administrator

# Disable Windows Update
net stop wuauserv
# Or: Set-Service -Name wuauserv -StartupType Disabled

# Disable Windows Defender real-time scanning (temporarily)
Set-MpPreference -DisableRealtimeMonitoring $true

# Close unnecessary background apps
Get-Process | Where-Object {$_.Name -like "OneDrive", "Teams", "Discord"} | Stop-Process -Force

# Clear pagefile and temporary files
cleanmgr

# Set power plan to High Performance
powercfg /setactive 8c5e7fda-e8bf-45a6-a6cc-4b8b65385cc0
```

### 4. Configuration

Edit `config.json` to customize:

```json
{
  "squad_samples": 250,  // Number of SQuAD examples (use 50 for quick test)
  "output_dir": "./results",
  "windows_specific": {
    "disable_windows_update": true,
    "disable_defender": true
  }
}
```

## Running Benchmarks

### Quick Test (30 minutes)
```bash
python run_benchmark.py --quick-test
```

### Full Benchmark Suite (8-12 hours)
```bash
python run_benchmark.py
```

### Specific Models Only
Edit `config.json` to list only the models you want, or adjust `run_benchmark.py` / orchestrator args if extended.

### Specific Runtimes Only
Edit `config.json` `"runtimes"` (e.g. `["llama_cpp", "pytorch"]`).

## Output Structure

```
results/
├── YYYYMMDD_HHMMSS/
│   ├── experiment_results.jsonl          # Raw experiment results
│   ├── summary_report.txt                # Summary statistics
│   ├── decision_matrix.txt               # Deployment recommendations
│   ├── results.csv                       # CSV export for analysis
│   ├── squad_dataset_cache.json          # Cached dataset for reproducibility
│   └── benchmark_YYYYMMDD_HHMMSS.log    # Detailed logs
```

## Model Quantization Guide

### Creating GGUF Models (for llama.cpp)

```bash
# Install conversion tools
pip install llama-cpp-python

# Convert HuggingFace model to GGUF
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# This would be a complex process
# For now, use pre-quantized models from HuggingFace:
# - TheBloke/Qwen2.5-1.5B-Instruct-GGUF
# - TheBloke/Mistral-7B-Instruct-v0.3-GGUF
```

### Creating GPTQ Models

```bash
# Install GPTQ tools
pip install auto-gptq

# This is complex - recommend using pre-quantized models:
# - TheBloke/Qwen2.5-1.5B-Instruct-GPTQ
# - TheBloke/Mistral-7B-Instruct-v0.3-GPTQ
```

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Ollama Connection Error
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, test connection
curl http://localhost:11434/api/tags
```

### Out of Memory
- Reduce `squad_samples` in config.json
- Use smaller models first (1.5B instead of 7B)
- Use quantized versions (INT8, Q4 instead of FP16)
- Close other applications

### Runtime Not Found
```bash
# Verify all runtimes are installed
python -c "from llama_cpp import Llama; print('llama.cpp OK')"
python -c "import onnxruntime; print('ONNX OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import requests; print('Ollama OK')"
```

## Performance Tips

1. **Cold-start tests**: Run in the morning after reboot for consistent cold-start metrics
2. **Warm-start tests**: Allow model to stay loaded between runs
3. **Single-thread**: Disable other CPU-intensive tasks
4. **Monitor temps**: Use HWInfo64 to watch GPU/CPU temperature (should be <80°C)
5. **Power**: Ensure stable power supply (no power saving mode)

## Analyzing Results

```bash
# After benchmark completes
python -c "from results_analyzer import generate_all_reports; generate_all_reports(Path('results/YYYYMMDD_HHMMSS'))"

# View reports
cat results/YYYYMMDD_HHMMSS/summary_report.txt
cat results/YYYYMMDD_HHMMSS/decision_matrix.txt

# Analyze in Excel
# Open results/YYYYMMDD_HHMMSS/results.csv
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| TTFT seems high | Check if model is still loading; increase warm-up runs |
| Memory spike | Normal during prefill phase; check peak not limit |
| Inconsistent results | Disable CPU frequency scaling: `powercfg /s 8c5e7fda-e8bf-45a6-a6cc-4b8b65385cc0` |
| GPU underutilized | Try smaller batch size or shorter prompts |

## Next Steps

1. Run quick test to verify setup: `python run_benchmark.py --quick-test`
2. Review results in `results/*/summary_report.txt`
3. Check decision matrix for recommendations
4. Run full benchmark overnight
5. Export results and analyze in Excel/Python

## Citation

When using this benchmark, please cite:
```
@inproceedings{chakaravarthi2025windows,
  title={Benchmarking Local Small Language Model Inference on Windows: Runtime, Quantization, and Deployment Trade-offs},
  author={Chakaravarthi, Ruban and Premkumar, Prithvikiran},
  booktitle={CSCI 5922, University of Colorado Boulder},
  year={2025}
}
```
