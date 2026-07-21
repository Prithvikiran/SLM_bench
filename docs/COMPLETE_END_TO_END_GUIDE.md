# COMPLETE END-TO-END PROJECT GUIDE
## Windows Local LLM Inference Benchmarking Suite

---

## TABLE OF CONTENTS
1. [Phase 1: Preparation](#phase-1-preparation)
2. [Phase 2: Environment Setup](#phase-2-environment-setup)
3. [Phase 3: Validation](#phase-3-validation)
4. [Phase 4: Configuration](#phase-4-configuration)
5. [Phase 5: Running Experiments](#phase-5-running-experiments)
6. [Phase 6: Analysis & Results](#phase-6-analysis--results)
7. [Phase 7: Documentation & Publication](#phase-7-documentation--publication)
8. [Troubleshooting](#troubleshooting)

---

# PHASE 1: PREPARATION
## (Time: 30 minutes - before you touch anything)

### Step 1.1: Understand What You're Building
**Goal:** Know what this project does

**Action:** Read these documents in order
```
1. START_HERE.md (5 min)
2. README.md (10 min)
3. PROJECT_SUMMARY.md (10 min)
```

**What to learn:**
- This benchmarks LLM inference on Windows
- Tests 5 models, 4 runtimes, 3 quantizations
- Measures both speed (TTFT, TPS) and quality (F1 score)
- Generates decision matrix for deployment

### Step 1.2: Understand Your System Requirements
**Goal:** Know if your hardware can run this

**Read:** SETUP_GUIDE.md - "System Requirements" section

**Minimum Requirements:**
- Windows 10 or 11
- 16GB RAM (32GB recommended)
- 100GB free storage (for models)
- Intel i5/i7 or AMD Ryzen 5/7 (8+ cores)
- Optional: NVIDIA GPU (RTX 4060 or better) for acceleration

**Action:** Check your system
```bash
# Windows - Open PowerShell and run:
Get-ComputerInfo | Select-Object CsProcessors, CsTotalPhysicalMemory

# Check free disk space
Get-Volume | Where-Object {$_.DriveLetter -eq 'C'}
```

### Step 1.3: Understand the Timeline
**Goal:** Know how long this takes

| Phase | Activity | Time |
|-------|----------|------|
| Setup | Environment, dependencies | 1-2 hours |
| Validation | Quick test | 5 minutes |
| Config | Customize if needed | 15 minutes |
| Full Benchmark | All experiments | 8-12 hours (overnight) |
| Analysis | Review results | 30 minutes |
| **Total** | **End to end** | **~12 hours** |

**Action:** Plan when to run
- Setup today (30 min active work)
- Validation today (quick)
- Full benchmark tonight (overnight, no active work)
- Analysis tomorrow morning (30 min)

### Step 1.4: Review Your Budget
**Goal:** Understand computational costs

**GPU Energy**: RTX 4060 benchmark = ~80W continuous for 12 hours = ~1 kWh
**Estimated cost**: $0.10-0.30 USD depending on electricity rates
**No cloud costs**: Everything runs locally on your machine

**Action:** None needed, just awareness

---

# PHASE 2: ENVIRONMENT SETUP
## (Time: 1-2 hours)

### Step 2.1: Install Python and Visual C++
**Goal:** Get base tools installed

**Action 2.1.1:** Install Python 3.10+
```bash
# Download from https://www.python.org/downloads/
# Get Python 3.11 or 3.12 (latest stable)
# During install: CHECK "Add Python to PATH"
# Click "Install Now"

# Verify installation
python --version
# Should show: Python 3.11.x or higher
```

**Action 2.1.2:** Install Visual C++ Build Tools (for CUDA)
```bash
# Only needed if you have NVIDIA GPU
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Install "Desktop development with C++"
```

**Verify:**
```bash
python -c "import sys; print(f'Python {sys.version}')"
```

### Step 2.2: Create Virtual Environment
**Goal:** Isolated Python environment for this project

**Action 2.2.1:** Navigate to project directory
```bash
# Open PowerShell or Command Prompt
# Navigate to where you put the code
cd C:\your\project\path\
# (Replace with actual path)
```

**Action 2.2.2:** Create virtual environment
```bash
python -m venv benchmark_env
```

**Action 2.2.3:** Activate it
```bash
# On Windows:
benchmark_env\Scripts\activate

# You should see (benchmark_env) in your prompt
```

**Verify:**
```bash
python -c "import sys; print(sys.prefix)"
# Should show: C:\your\project\path\benchmark_env
```

### Step 2.3: Install Python Dependencies
**Goal:** Get all required packages

**Action 2.3.1:** Upgrade pip
```bash
python -m pip install --upgrade pip
```

**Action 2.3.2:** Install PyTorch with CUDA (GPU acceleration)
```bash
# For NVIDIA GPU (RTX 3060+, RTX 4060+, etc.):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
# Should show: True (if GPU) or False (if CPU)
```

**Action 2.3.3:** Install core dependencies
```bash
pip install transformers datasets psutil requests
```

**Action 2.3.4:** Install runtime-specific packages
```bash
# llama.cpp Python bindings
pip install llama-cpp-python

# ONNX Runtime (for GPU, use onnxruntime-gpu for CPU)
pip install onnxruntime-gpu

# Quantization support
pip install bitsandbytes peft

# Analysis tools
pip install pandas numpy
```

**Verify all installed:**
```bash
python -c "import torch, transformers, datasets, psutil, requests, llama_cpp, onnxruntime, pandas; print('✅ All packages installed')"
```

### Step 2.4: Install Ollama (Optional but Recommended)
**Goal:** Setup Ollama runtime

**Action 2.4.1:** Download and install
```bash
# Go to https://ollama.com
# Download Windows installer
# Run installer (accepts all defaults)
```

**Action 2.4.2:** Verify installation
```bash
# In new PowerShell, run:
ollama --version
# Should show: ollama version X.X.X
```

**Action 2.4.3:** Start Ollama service (before running benchmark)
```bash
# Run this before benchmark:
ollama serve

# You should see:
# "Listening on 127.0.0.1:11434"
# Leave this running while benchmark uses Ollama
```

---

# PHASE 3: VALIDATION
## (Time: 5 minutes)

### Step 3.1: Verify Everything Works
**Goal:** Quick test that all dependencies work

**Action 3.1.1:** Activate your environment
```bash
# If not already activated:
benchmark_env\Scripts\activate
```

**Action 3.1.2:** Run quick validation
```bash
python run_benchmark.py --quick-test
```

**Expected output:**
```
================================================================================
QUICK VALIDATION TEST
================================================================================

1. Testing imports...
✓ All imports successful

2. Checking system configuration...
  OS: Windows 11 22621
  Python: 3.11.x
  GPU: NVIDIA GeForce RTX 4060

3. Testing SQuAD task handler...
✓ Task handler OK

4. Testing metrics collector...
✓ Metrics collector OK

5. Testing runtime factory...
✓ pytorch available
✓ onnx_runtime available

================================================================================
QUICK TEST PASSED - Ready for full benchmark
================================================================================
```

**If you see ✓ everywhere:** Success! Go to Phase 4.

**If you see errors:** Go to [Troubleshooting](#troubleshooting) section.

---

# PHASE 4: CONFIGURATION
## (Time: 15 minutes)

### Step 4.1: Review Default Configuration
**Goal:** Understand what will be tested

**Action 4.1.1:** Open config.json
```bash
# In VS Code or Notepad:
code config.json
# or
notepad config.json
```

**What you'll see:**
```json
{
  "models": {
    "small": ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "Phi-3.5-mini-Instruct"],
    "medium": ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"]
  },
  "runtimes": ["llama_cpp", "ollama", "pytorch", "onnx_runtime"],
  "quantizations": ["fp16", "int8", "q4"],
  "workloads": ["interactive_chat", "structured_output", "long_context"],
  "squad_samples": 250
}
```

### Step 4.2: Customize Configuration (Optional)
**Goal:** Adjust for your system if needed

**Common Customizations:**

**Option A: Quick Test (30 min instead of 8 hours)**
```json
// In config.json, change:
"squad_samples": 50,          // Reduce from 250
"trials_per_config": 1,        // Reduce from 3
"runtimes": ["pytorch"],       // Test only PyTorch
```

**Option B: Single Model Test**
```json
"models": {
  "small": ["Qwen2.5-1.5B-Instruct"],
  "medium": []
}
```

**Option C: CPU Only (no GPU)**
```json
"runtimes": ["llama_cpp", "pytorch"],  // Skip ollama if not using GPU
```

**Action 4.2.1:** Save config.json
```bash
# After making any changes:
# Ctrl+S (save and close)
```

### Step 4.3: Understand Output Structure
**Goal:** Know where results will be saved

**Results location:**
```
results/
├── 20250415_143022/              # YYYYMMDD_HHMMSS
│   ├── experiment_results.jsonl  # Raw data (1 line per experiment)
│   ├── summary_report.txt        # Key findings
│   ├── decision_matrix.txt       # Recommendations
│   ├── results.csv               # For Excel analysis
│   └── benchmark_log.log         # Detailed logs
```

**You will analyze:** `summary_report.txt` and `decision_matrix.txt`

---

# PHASE 5: RUNNING EXPERIMENTS
## (Time: 5 minutes setup + 8-12 hours runtime)

### Step 5.1: Pre-Benchmark Checks
**Goal:** Ensure system is ready

**Action 5.1.1:** Optimize Windows for benchmarking
```powershell
# Run as Administrator:

# Disable Windows Update
net stop wuauserv

# Disable Defender real-time scanning (temporary)
Set-MpPreference -DisableRealtimeMonitoring $true

# Close unnecessary apps (Teams, Discord, OneDrive, etc.)
Get-Process | Where-Object {$_.Name -like "*Teams*", "*Discord*"} | Stop-Process -Force
```

**Action 5.1.2:** Set High Performance power plan
```powershell
# Run as Administrator:
powercfg /setactive 8c5e7fda-e8bf-45a6-a6cc-4b8b65385cc0
```

**Action 5.1.3:** Start Ollama (if using)
```bash
# In separate PowerShell window:
ollama serve
# Keep this running while benchmark uses Ollama
```

### Step 5.2: Run Full Benchmark
**Goal:** Execute all experiments

**Action 5.2.1:** Ensure environment activated
```bash
benchmark_env\Scripts\activate
```

**Action 5.2.2:** Start benchmark
```bash
# For full benchmark (recommended - run overnight):
python run_benchmark.py

# OR for quick test (30 min):
python run_benchmark.py --quick-test

# OR specific models/runtimes:
python run_benchmark.py --models "Qwen2.5-7B-Instruct" --runtimes pytorch
```

**What you'll see:**
```
================================================================================
STARTING WINDOWS LLM INFERENCE BENCHMARK
================================================================================

[1/6] Verifying system environment...
✓ OS: Windows 11
✓ Python: 3.11
✓ GPU: RTX 4060

[2/6] Loading SQuAD v2.0 dataset...
Loaded 250 SQuAD v2 examples

[3/6] Preparing models and quantizations...
Preparing Qwen2.5-1.5B-Instruct...
Preparing Qwen2.5-7B-Instruct...
...

[4/6] Setting up inference runtimes...

[5/6] Running benchmarking experiments...

Experiment: Qwen2.5-1.5B | pytorch | fp16 | interactive_chat | Trial 1
  Sample 1/20: TTFT=45.3ms, TPS=120.5
  Sample 2/20: TTFT=42.1ms, TPS=125.3
  ...

[6/6] Analyzing results and generating reports...
Generating summary_report.txt...
Generating decision_matrix.txt...
Generating results.csv...

================================================================================
BENCHMARK COMPLETED SUCCESSFULLY (3.2 hours)
Results saved to: results/20250415_143022
================================================================================
```

### Step 5.3: Monitor Progress (Optional)
**Goal:** Watch what's happening

**Action 5.3.1:** Check logs in real-time
```bash
# In another PowerShell:
tail -f results/*/benchmark_*.log

# Or open directly:
Get-Content results/20250415_143022/benchmark_*.log
```

**Action 5.3.2:** Monitor system resources
```bash
# Open Task Manager: Ctrl+Shift+Esc
# Watch: GPU usage, Memory, CPU
# Should see GPU at 80-90% utilization during benchmarks
```

### Step 5.4: What NOT to Do During Benchmark
**Action:** Avoid these during runtime
- ❌ Don't restart your computer
- ❌ Don't close PowerShell windows
- ❌ Don't run other GPU-intensive programs
- ❌ Don't disable GPU drivers
- ❌ Don't unplug internet (if using Ollama)

---

# PHASE 6: ANALYSIS & RESULTS
## (Time: 30 minutes)

### Step 6.1: Locate Results
**Goal:** Find your benchmark results

**Action 6.1.1:** After benchmark completes
```bash
# Results are in:
cd results
ls -la
# You'll see: 20250415_143022 (or your timestamp)

cd 20250415_143022
ls -la
# Files present:
# - experiment_results.jsonl
# - summary_report.txt
# - decision_matrix.txt
# - results.csv
```

### Step 6.2: View Summary Report
**Goal:** See key statistics

**Action 6.2.1:** Open summary report
```bash
# View in PowerShell:
Get-Content summary_report.txt

# Or open in Notepad:
notepad summary_report.txt
```

**You'll see:**
```
================================================================================
WINDOWS LLM INFERENCE BENCHMARKING REPORT
================================================================================

OVERALL STATISTICS
─────────────────────────────────────────────────────────────────
Total experiments: 180
Models tested: 5
Runtimes tested: 4
Quantizations tested: 3

TASK PERFORMANCE (SQuAD v2.0)
─────────────────────────────────────────────────────────────────
Model                           F1      Exact Match
Qwen2.5-1.5B                   0.812   0.684
Qwen2.5-7B                     0.847   0.721
Mistral-7B                     0.823   0.698
Phi-3.5-mini                   0.798   0.671
Llama-3.2-1B                   0.805   0.679

RUNTIME PERFORMANCE
─────────────────────────────────────────────────────────────────
Runtime         TTFT (ms)  Throughput (TPS)
pytorch         78.3       52.1
onnx_runtime    65.2       58.3
ollama          180.5      45.2
llama_cpp       42.1       125.3

QUANTIZATION IMPACT
─────────────────────────────────────────────────────────────────
Quantization    F1       Memory (MB)
FP16            0.845    13850
INT8            0.841    7200
Q4              0.832    4500
```

**What this tells you:**
- Best F1 score: Qwen2.5-7B (0.847)
- Fastest TTFT: llama.cpp (42ms)
- Best throughput: llama.cpp (125 tokens/sec)
- Best balance: ONNX Runtime (65ms TTFT, 58 TPS)

### Step 6.3: View Decision Matrix
**Goal:** See deployment recommendations

**Action 6.3.1:** Open decision matrix
```bash
notepad decision_matrix.txt
```

**You'll see:**
```
================================================================================
WINDOWS LLM DEPLOYMENT DECISION MATRIX
================================================================================

SCENARIO 1: Interactive Chat (Low Latency Critical)
─────────────────────────────────────────────────────────────────
Model                Runtime        Quant  TTFT(ms)  F1     Memory(MB)
Qwen2.5-1.5B        llama.cpp      Q4     20ms      0.81   650
Phi-3.5-mini        llama.cpp      Q4     28ms      0.79   580
Llama-3.2-1B        llama.cpp      Q4     25ms      0.80   620

SCENARIO 2: Batch Processing (Throughput Critical)
─────────────────────────────────────────────────────────────────
Model                Runtime        Quant  TPS       F1     Memory(MB)
Qwen2.5-7B          pytorch        FP16   52.1      0.85   14200
Mistral-7B          pytorch        FP16   48.3      0.82   13900
Phi-3.5-mini        onnx_runtime   INT8   58.3      0.79   7200

SCENARIO 3: Quality First (High Accuracy)
─────────────────────────────────────────────────────────────────
Model                Runtime        Quant  F1        TTFT   Memory(MB)
Qwen2.5-7B          pytorch        FP16   0.847     78ms   14200
Mistral-7B          pytorch        FP16   0.823     85ms   13900
Qwen2.5-7B          onnx_runtime   FP16   0.845     65ms   13800

SCENARIO 4: Memory Constrained (Small Devices)
─────────────────────────────────────────────────────────────────
Model                Runtime        Quant  Memory    F1     TTFT
Phi-3.5-mini        llama.cpp      Q4     580MB     0.79   28ms
Qwen2.5-1.5B        llama.cpp      Q4     650MB     0.81   20ms
Llama-3.2-1B        llama.cpp      Q4     620MB     0.80   25ms
```

**How to use this:**
- For your use case, find the matching scenario
- Choose the "recommended" row
- Use those settings (model, runtime, quantization)
- Deploy with confidence knowing it's optimal for your needs

### Step 6.4: Analyze Raw Data in Excel
**Goal:** Deep-dive analysis

**Action 6.4.1:** Open CSV in Excel
```bash
# Right-click results.csv
# Open with Excel
```

**What you can do:**
- Sort by F1 (see best quality)
- Sort by TTFT (see fastest)
- Sort by Memory (see smallest)
- Create pivot tables
- Make graphs/charts
- Filter by runtime/quantization

### Step 6.5: Export Results
**Goal:** Share or keep results

**Action 6.5.1:** Copy important files
```bash
# Copy results to safe location:
Copy-Item results\20250415_143022\summary_report.txt -Destination C:\MyResults\
Copy-Item results\20250415_143022\decision_matrix.txt -Destination C:\MyResults\
Copy-Item results\20250415_143022\results.csv -Destination C:\MyResults\
```

---

# PHASE 7: DOCUMENTATION & PUBLICATION
## (Time: Variable, depends on your use case)

### Step 7.1: Write Your Report
**Goal:** Document findings

**Action 7.1.1:** Create report document
```bash
# Open Word or Google Docs
# Create report with sections:
```

**Report Template:**
```
# LLM Inference Benchmarking on Windows - Results Report

## Executive Summary
[2-3 sentences of key findings]

## Hardware Setup
- OS: Windows 11
- GPU: RTX 4060
- CPU: Intel i7-13700K
- RAM: 32GB
- Date: [date]

## Methodology
- Models: 5 (Qwen, Llama, Mistral, Phi)
- Runtimes: 4 (llama.cpp, Ollama, PyTorch, ONNX)
- Task: SQuAD v2.0 Extractive QA
- Samples: 250 examples

## Key Findings
1. [Fastest runtime for latency]
2. [Best F1 score]
3. [Best memory efficiency]
4. [Best overall balance]

## Recommendations by Scenario
[Copy from decision_matrix.txt]

## Raw Results
[Copy tables from summary_report.txt]

## Conclusion
[Summary of implications]
```

### Step 7.2: Create Visualization
**Goal:** Charts for presentations

**Action 7.2.1:** In Excel, create charts
```
1. Bar chart: F1 score by model
2. Line chart: TTFT by runtime
3. Scatter plot: Memory vs F1 (efficiency frontier)
4. Comparison matrix: Runtime x Quantization
```

### Step 7.3: Share Findings
**Goal:** Communicate results

**Where to share:**
- Email to team with summary_report.txt + decision_matrix.txt
- SharePoint/OneDrive for results.csv
- GitHub if publishing code
- Research paper if academic publication
- Blog post for public audience

### Step 7.4: Publication (If Academic)
**Goal:** Prepare for academic submission

**Files to include:**
```
1. experiment_results.jsonl          # Raw data for reproducibility
2. config.json                       # Exact config used
3. benchmark_log.log                 # Full execution logs
4. summary_report.txt                # Key metrics
5. decision_matrix.txt               # Recommendations
6. results.csv                       # For external analysis
7. Your written paper                # The research
```

**Citation format:**
```bibtex
@inproceedings{chakaravarthi2025windows,
  title={Benchmarking Local Small Language Model Inference on Windows: 
         Runtime, Quantization, and Deployment Trade-offs},
  author={Chakaravarthi, Ruban and Premkumar, Prithvikiran},
  booktitle={CSCI 5922, University of Colorado Boulder},
  year={2025}
}
```

---

# TROUBLESHOOTING
## (Reference when things go wrong)

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Cause:** PyTorch not installed

**Solution:**
```bash
# Ensure venv is activated
benchmark_env\Scripts\activate

# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

### Problem: "CUDA is not available" (have NVIDIA GPU)

**Cause:** CUDA drivers or PyTorch GPU version mismatch

**Solution:**
```bash
# Check NVIDIA drivers
nvidia-smi
# Should show GPU info

# If not, install drivers from https://www.nvidia.com/Download/index.aspx

# Reinstall PyTorch with correct CUDA:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "llama-cpp-python installation fails"

**Cause:** Visual C++ build tools missing

**Solution:**
```bash
# Install Visual C++ Build Tools:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++"

# Then retry:
pip install llama-cpp-python --force-reinstall
```

### Problem: "Ollama connection refused"

**Cause:** Ollama service not running

**Solution:**
```bash
# Open new PowerShell window and run:
ollama serve

# OR start Ollama service:
# Go to Start Menu → Search "Services"
# Find "Ollama" → Right-click → Start
```

### Problem: "Out of memory" error

**Cause:** GPU/CPU memory exhausted

**Solution:**
```json
// In config.json, reduce:
"squad_samples": 50,      // Down from 250
"trials_per_config": 1,   // Down from 3
"models": {
  "small": ["Qwen2.5-1.5B-Instruct"],  // Test one model
  "medium": []
}
```

### Problem: "TTFT is very high (>500ms)"

**Cause:** Model still loading, Windows background tasks

**Solution:**
1. Run benchmark again (first run slower due to caching)
2. Close other applications
3. Disable Windows Update: `net stop wuauserv`
4. Check Task Manager for CPU/GPU usage

### Problem: "Results directory is empty"

**Cause:** Benchmark crashed silently

**Solution:**
```bash
# Check logs:
cat results/*/benchmark_*.log

# Run with debugging:
python run_benchmark.py 2>&1 | Tee-Object -FilePath debug.log

# Check for errors and report to project
```

### Problem: "GPU memory leak - memory keeps growing"

**Cause:** Model not properly unloading

**Solution:**
```bash
# This is normal between experiments
# GPU memory is recycled automatically

# If it doesn't recover:
# Restart PowerShell and benchmark
```

---

# QUICK REFERENCE COMMANDS

```bash
# Activate environment
benchmark_env\Scripts\activate

# Deactivate environment
deactivate

# Quick validation
python run_benchmark.py --quick-test

# Full benchmark
python run_benchmark.py

# Run specific models
python run_benchmark.py --models "Qwen2.5-1.5B-Instruct" "Qwen2.5-7B-Instruct"

# Run specific runtimes
python run_benchmark.py --runtimes pytorch onnx_runtime

# View results
notepad results\YYYYMMDD_HHMMSS\summary_report.txt

# View decision matrix
notepad results\YYYYMMDD_HHMMSS\decision_matrix.txt

# Open in Excel
excel results\YYYYMMDD_HHMMSS\results.csv

# Check latest results (list by timestamp)
ls results -Directory | Sort-Object Name -Descending | Select-Object -First 1
```

---

# NEXT STEPS AFTER COMPLETION

1. **Share Results** with team/supervisor
2. **Write Paper** using template from Phase 7.1
3. **Create Presentation** using graphs from Phase 7.2
4. **Submit Findings** to academic venues (if applicable)
5. **Deploy Optimal Config** from decision_matrix in production
6. **Archive Results** for reproducibility

---

# SUPPORT & HELP

If you get stuck:

1. **Check SETUP_GUIDE.md** - Most issues documented
2. **Review troubleshooting** above
3. **Check benchmark_*.log** - Detailed error messages
4. **Run quick-test** - Validates environment
5. **Contact support** with error message + log file

---

**You're ready! Follow phases 1-6 and you'll have complete benchmark results.**

**Estimated total time: 12-14 hours (mostly automated overnight)**

Good luck! 🚀
