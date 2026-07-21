# TASK DOCUMENTATION
## Windows Local LLM Inference Benchmarking

---

## TABLE OF CONTENTS
1. [What Exactly Is Done](#what-exactly-is-done)
2. [Task Definition](#task-definition)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Metrics](#metrics)
6. [Experimental Design](#experimental-design)
7. [Data Flow](#data-flow)
8. [Output Format](#output-format)

---

# WHAT EXACTLY IS DONE

## High-Level Overview

**Goal:** Systematically measure and compare how different inference runtimes, quantization methods, and language models perform on Windows hardware when running the same NLP task.

**Scope:** Benchmark extractive question-answering inference (SQuAD v2.0) using multiple combinations of:
- 5 different language models (varying sizes: 1B to 7B parameters)
- 4 different inference runtimes (llama.cpp, Ollama, PyTorch, ONNX Runtime)
- 3 different quantization levels (FP16, INT8, Q4)
- 3 different workload patterns (interactive chat, structured output, long-context)

**Output:** A comprehensive benchmark report that recommends the optimal configuration for different deployment scenarios.

---

## What Gets Measured

For each configuration combination, the benchmark measures:

### System Performance
- Time to First Token (TTFT) - How long before first output
- Tokens Per Second (TPS) - Generation speed
- Memory Usage - Both CPU and GPU
- Cold Start Time - Model load + first inference
- Warm Start Time - Subsequent inferences

### Task Performance
- F1 Score - Token-level overlap with reference answer
- Exact Match Rate - Perfect match percentage
- JSON Validity - For structured output tasks

### What Gets Compared
Each metric is collected across:
- All 5 models
- All 4 runtimes
- All 3 quantization levels
- All 3 workload types
- Multiple trials (3-5 per configuration)

---

# TASK DEFINITION

## Primary Task: SQuAD v2.0 Extractive Question Answering

### What is SQuAD?
**SQuAD (Stanford Question Answering Dataset) v2.0** is a reading comprehension dataset where:
- You are given a **passage** (context)
- You are given a **question** about that passage
- You must provide the **exact span** from the passage that answers the question
- Some questions are unanswerable (answer not in passage)

### Example

```
CONTEXT:
"The Great Wall of China is a series of fortifications made of stone, brick, 
tamped earth and wood, built along the historical northern borders of China 
to protect the Chinese states and empires against the raids and invasions 
of various nomadic groups."

QUESTION:
"What was the Great Wall of China built with?"

EXPECTED ANSWER:
"stone, brick, tamped earth and wood"
```

### Why This Task?

1. **Reproducible** — Same task, same evaluation metric (F1 score)
2. **Standardized** — Official SQuAD evaluation script exists
3. **Practical** — Real use case (customer support, document QA)
4. **Moderate Complexity** — Tests both understanding and extraction
5. **Citable** — Results can be referenced in academic papers

---

## Three Workload Variants

The same SQuAD dataset is evaluated in three different workload patterns:

### Workload 1: Interactive Chat
**Purpose:** Measure latency-sensitive inference (chatbot use case)

**What happens:**
1. Short context passages (100-300 tokens)
2. Single question
3. Model generates answer
4. Measure: TTFT (most important), throughput, memory

**Relevance:** Customer support bots, real-time assistance

**Example:**
```
Input: "What is Python used for?"
Output: "Python is used for web development, data science, automation..."
Response Time: 45ms (TTFT) + 200ms (full response) = ~250ms total
```

### Workload 2: Structured Output
**Purpose:** Test ability to generate valid JSON responses

**What happens:**
1. Generate answers in JSON format
2. Validate JSON structure
3. Measure: JSON validity rate, F1 score, memory

**Relevance:** API responses, programmatic interfaces

**Example:**
```
Input: "Who invented the telephone?"
Expected Output: {
  "answer": "Alexander Graham Bell",
  "confidence": "high"
}
Metric: Is valid JSON? (yes/no)
```

### Workload 3: Long-Context
**Purpose:** Test ability to handle large documents (multi-passage reasoning)

**What happens:**
1. Concatenate 5-10 SQuAD passages (2k-4k tokens)
2. Question requires reasoning across passages
3. Measure: Throughput, memory during decode phase, F1 score

**Relevance:** Document summarization, long-form QA

**Example:**
```
Input: Multiple Wikipedia paragraphs + synthesis question
"Summarize the history of the internet from the context above"
Output: Multi-paragraph answer
Memory Peak: During decoding phase
```

---

# DATASET

## SQuAD v2.0 Dataset Details

### Source
- **Official Source:** https://rajpurkar.github.io/SQuAD-explorer/
- **HuggingFace Dataset:** `squad_v2`
- **License:** CC BY-SA 4.0

### Dataset Statistics (Full)
```
Total Questions: 150,000+
Total Passages: 35,000+
Training Set: ~130,000 questions
Validation Set: ~12,000 questions
Unanswerable Questions: ~5,200 (v2.0 addition)
```

### Dataset Usage in This Project

**Sampling Strategy:**
```
Full SQuAD v2.0 validation set
         ↓
Random sample 250 examples
         ↓
Use same 250 for all experiments
         ↓
Ensures reproducibility
```

**Why 250 examples?**
- Large enough for statistical significance
- Small enough to complete benchmarks in reasonable time
- Balances between variance and execution time

### Data Preparation

**Step 1: Download**
```python
from datasets import load_dataset
dataset = load_dataset('squad_v2')
```

**Step 2: Extract validation set**
```python
validation_set = dataset['validation']  # 12,442 examples
```

**Step 3: Sample reproducibly**
```python
import random
random.seed(42)
sampled = random.sample(list(validation_set), 250)
```

**Step 4: Cache for reproducibility**
```json
{
  "contexts": ["The Great Wall...", ...],
  "questions": ["What was...?", ...],
  "answers": [{"text": "stone, brick...", "answer_start": 78}, ...],
  "ids": ["56be4db0acb8001400a502ec", ...]
}
```

### Data Format

Each SQuAD example contains:

```json
{
  "id": "56be4db0acb8001400a502ec",
  "context": "The Great Wall of China is a series of fortifications...",
  "question": "What was the Great Wall of China built with?",
  "answers": [
    {
      "text": "stone, brick, tamped earth and wood",
      "answer_start": 78
    },
    {
      "text": "stone, brick, tamped earth and wood",
      "answer_start": 78
    }
  ],
  "is_impossible": false
}
```

### Train/Test Split

```
All 250 examples used for:
- Interactive Chat: 20 examples per trial × 3 trials = 60 examples
- Structured Output: 20 examples per trial × 3 trials = 60 examples
- Long-Context: 20 examples per trial × 3 trials = 60 examples

Total: ~180 unique evaluations per model configuration
```

---

# MODELS

## 5 Models Across 2 Size Tiers

### Small Tier (1-3.8B parameters)

#### Model 1: Qwen2.5-1.5B-Instruct
```
HuggingFace ID: Qwen/Qwen2.5-1.5B-Instruct
Parameters: 1.5 Billion
Context Window: 32,768 tokens (32k)
Training Data: 7 Trillion tokens
Strengths: 
  - Excellent SQuAD performance
  - Strong quantization support
  - Fast inference
Architecture:
  - Transformer-based
  - Rotary embeddings
  - GQA (Grouped Query Attention)
  
Quantization Support: GGUF (4-bit, 5-bit), GPTQ, AWQ
Memory Requirements:
  - FP16: ~3 GB
  - INT8: ~1.5 GB
  - Q4: ~500-800 MB
```

#### Model 2: Llama-3.2-1B-Instruct
```
HuggingFace ID: meta-llama/Llama-3.2-1B-Instruct
Parameters: 1 Billion
Context Window: 8,192 tokens (8k)
Training Data: 15 Trillion tokens (Meta)
Strengths:
  - Recent model from Meta
  - Strong instruction following
  - Different architecture than Qwen (for comparison)
Architecture:
  - Transformer with RoPE
  - Multi-head attention
  - 16 attention heads

Quantization Support: GGUF, GPTQ, AWQ
Memory Requirements:
  - FP16: ~2 GB
  - INT8: ~1 GB
  - Q4: ~400-600 MB
```

#### Model 3: Phi-3.5-mini-Instruct (NOVEL)
```
HuggingFace ID: microsoft/Phi-3.5-mini-instruct
Parameters: 3.8 Billion
Context Window: 8,192 tokens (8k)
Training Data: 39 Billion tokens (Microsoft)
Strengths:
  - Microsoft-optimized
  - Excellent for Windows + ONNX
  - Outstanding latency for size
  - Strong on SQuAD despite being small
Architecture:
  - Optimized transformer
  - Multi-head attention
  - Flash attention support

Quantization Support: GGUF, GPTQ, ONNX INT8
Memory Requirements:
  - FP16: ~7.5 GB
  - INT8: ~4 GB
  - Q4: ~1.5-2 GB
  
Why Novel: First systematic Windows benchmark for Phi
```

### Medium Tier (7B parameters)

#### Model 4: Qwen2.5-7B-Instruct
```
HuggingFace ID: Qwen/Qwen2.5-7B-Instruct
Parameters: 7 Billion
Context Window: 128,000 tokens (128k) — LONG CONTEXT!
Training Data: 18 Trillion tokens
Strengths:
  - State-of-the-art for 7B
  - Excellent F1 on SQuAD (84.7%)
  - Same family as 1.5B (fair comparison)
  - Supports long context
Architecture:
  - Same as 1.5B (scaled up)
  - GQA attention
  - RoPE embeddings

Quantization Support: GGUF, GPTQ, AWQ
Memory Requirements:
  - FP16: ~14 GB
  - INT8: ~7.5 GB
  - Q4: ~2.5-3 GB
```

#### Model 5: Mistral-7B-Instruct-v0.3
```
HuggingFace ID: mistralai/Mistral-7B-Instruct-v0.3
Parameters: 7 Billion
Context Window: 32,768 tokens (32k)
Training Data: 7 Trillion tokens
Strengths:
  - Different MQA attention (affects performance differently)
  - Widely used in 2024-2025
  - Strong F1 on SQuAD (82.3%)
Architecture:
  - Transformer
  - Multi-Query Attention (MQA) — different from Qwen!
  - Grouped query attention variant

Quantization Support: GGUF, GPTQ, AWQ
Memory Requirements:
  - FP16: ~14 GB
  - INT8: ~7.5 GB
  - Q4: ~2.5-3 GB
  
Why Different: Different attention mechanism tests if findings generalize
```

## Model Justification

| Model | Why Chosen | What It Tests |
|-------|-----------|--------------|
| Qwen-1.5B | Same family as 7B | Scale effects within same architecture |
| Qwen-7B | SOTA for size | Baseline quality (84.7% F1) |
| Llama-1B | Different family | Generalization across architectures |
| Mistral-7B | Different attention | Architecture matters (MQA vs GQA) |
| Phi-3.5-mini | Microsoft-native | Windows + ONNX optimization |

---

# METRICS

## System Metrics

### 1. Time to First Token (TTFT)
**Definition:** Milliseconds from request submission to first token output

**Formula:**
```
TTFT = first_token_timestamp - request_start_timestamp
```

**Why it matters:**
- User perceives responsiveness
- Critical for real-time applications
- Dominated by prefill (context processing)

**Range for benchmarks:**
- llama.cpp Q4: 20-50ms (fastest)
- PyTorch FP16: 60-150ms (slowest on GPU)
- ONNX INT8: 40-80ms (optimized)

**Example:**
```
Request at: 0ms
First token at: 45ms
→ TTFT = 45ms
```

### 2. Tokens Per Second (TPS)
**Definition:** How many tokens generated per second during decode phase

**Formula:**
```
TPS = output_tokens / (final_token_timestamp - first_token_timestamp)
```

**Why it matters:**
- User waits proportionally (longer responses = longer waits)
- Affects throughput for batch processing
- GPU utilization proxy

**Range for benchmarks:**
- llama.cpp: 100-150 tokens/sec
- PyTorch: 40-60 tokens/sec (context dependent)
- ONNX: 50-70 tokens/sec

**Example:**
```
Generated 64 tokens in 0.8 seconds
→ TPS = 64 / 0.8 = 80 tokens/sec
```

### 3. Peak Memory Usage
**Definition:** Maximum RAM or VRAM consumed during inference

**Measured at:** End of prefill + early decode phase

**Why it matters:**
- Determines hardware minimum requirements
- Critical for edge devices
- Affects cost (cloud) or feasibility (local)

**Memory Components:**

**CPU Memory:**
- Model weights (depends on quantization)
- Activations (intermediate tensors)
- KV cache (stored attention values)

**GPU Memory:**
- Model weights on GPU
- Activations on GPU
- Attention KV cache on GPU

**Example:**
```
Model FP16 on GPU:
- Weights: 13 GB
- Activations + KV cache: 1.5 GB
→ Peak memory: ~14.5 GB
```

### 4. CPU Utilization
**Definition:** Percentage of CPU cores used during inference

**Measured:** Average during inference

**Why it matters:**
- GPU-bound vs CPU-bound detection
- Multi-core efficiency
- Thermal implications

**Range:**
- llama.cpp CPU-bound: 70-95% CPU
- PyTorch GPU-bound: 20-40% CPU
- ONNX GPU-bound: 15-30% CPU

### 5. Cold Start Metrics
**Definition:** Time to load model + first inference

**Breakdown:**
```
Total Cold Start = Model Load Time + First Inference Time

Example:
- Load model: 2.5 seconds (DLL loading, GPU transfer)
- First inference: 0.15 seconds (includes TTFT)
→ Total: 2.65 seconds
```

**Why it matters:**
- First user experiences this delay
- Important for infrequent queries
- Serverless deployment cost

---

## Task Metrics

### 1. F1 Score (Primary Quality Metric)
**Definition:** Token-level harmonic mean of precision and recall

**Formula:**
```
Precision = |predicted_tokens ∩ reference_tokens| / |predicted_tokens|
Recall = |predicted_tokens ∩ reference_tokens| / |reference_tokens|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Why it matters:**
- Standard SQuAD evaluation metric
- Reflects partial credit (not just exact matches)
- Published baselines exist

**Normalization:**
```
Before comparison:
1. Remove articles (a, an, the)
2. Remove punctuation
3. Lowercase everything
4. Collapse whitespace

Example:
Input: "The stone, brick, earth."
Normalized: "stone brick earth"
```

**Range:**
- Perfect answer: F1 = 1.0
- Half correct: F1 ≈ 0.5
- Wrong: F1 = 0.0

**Example:**
```
Reference: "stone brick tamped earth and wood"
Reference tokens: {stone, brick, tamped, earth, and, wood}

Prediction 1: "stone brick and tamped earth"
Predicted tokens: {stone, brick, and, tamped, earth}
Common: {stone, brick, tamped, earth, and} (5 tokens)
Precision = 5/5 = 1.0
Recall = 5/6 = 0.833
F1 = 2*(1.0*0.833)/(1.0+0.833) = 0.909

Prediction 2: "stone and brick"
Predicted tokens: {stone, and, brick}
Common: {stone, and, brick} (3 tokens)
Precision = 3/3 = 1.0
Recall = 3/6 = 0.5
F1 = 2*(1.0*0.5)/(1.0+0.5) = 0.667
```

### 2. Exact Match Rate
**Definition:** Percentage of questions where prediction matches reference exactly (after normalization)

**Formula:**
```
EM = (count of exact matches) / (total questions)
```

**Why it matters:**
- Strict evaluation
- Reflects when model "got it right"
- Published baseline available

**Example:**
```
Out of 20 questions:
- 15 exact matches
- 5 partial matches

EM = 15/20 = 75%
```

### 3. JSON Validity Rate (Structured Output Task Only)
**Definition:** Percentage of outputs that are valid, parseable JSON

**Why it matters:**
- API integration requires valid JSON
- Measures if model follows format instructions
- Impacts downstream automation

**Example:**
```
Output 1: {"answer": "Paris", "confidence": "high"}
→ Valid JSON ✓

Output 2: {"answer": "Paris" "confidence": "high"}
→ Invalid (missing comma) ✗

Validity Rate = 1/2 = 50%
```

---

## Metric Aggregation

### Per Trial
**Collected:** For each single inference
```json
{
  "ttft_ms": 45.3,
  "tps": 120.5,
  "peak_memory_mb": 6800,
  "f1": 0.87,
  "exact_match": 1,
  "json_valid": 1
}
```

### Per Configuration (across trials)
**Aggregated:** Across 3-5 trials of same config
```json
{
  "ttft_ms_mean": 46.2,
  "ttft_ms_median": 45.8,
  "ttft_ms_std": 1.2,
  
  "tps_mean": 119.3,
  "f1_mean": 0.853,
  "exact_match_mean": 0.95,
  "json_valid_rate": 0.98,
  
  "peak_memory_mb_max": 7100
}
```

### Cross-Configuration (decision matrix)
**Compared:** Across all combinations
```
Best F1: Qwen2.5-7B PyTorch FP16 (0.847)
Fastest TTFT: Qwen2.5-1.5B llama.cpp Q4 (22ms)
Most Efficient: Phi-3.5-mini ONNX INT8 (2.6GB memory, 0.79 F1)
```

---

# EXPERIMENTAL DESIGN

## Configuration Matrix

**Total Combinations:**
```
Models: 5
  × Runtimes: 4
  × Quantizations: 3
  × Workloads: 3
  × Trials: 3 (warm) + 1 (cold)
────────────────────────────
= ~200 total experiments
```

## Experiment Flow (Single Configuration)

**Configuration Example:** Qwen-7B + PyTorch + FP16 + Interactive Chat

```
┌─────────────────────────────┐
│ SETUP PHASE                 │
│ ├─ Load model (FP16)        │
│ ├─ Initialize metrics       │
│ └─ Start monitoring         │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│ WARM-START TRIAL (3x)       │
│ ├─ Sample 20 SQuAD examples │
│ ├─ For each example:        │
│ │  ├─ Generate prompt      │
│ │  ├─ Start TTFT timer     │
│ │  ├─ Run inference        │
│ │  ├─ Record metrics       │
│ │  ├─ Extract answer       │
│ │  └─ Evaluate vs reference│
│ └─ Aggregate stats          │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│ COLD-START TRIAL (1x)       │
│ ├─ Unload model             │
│ ├─ Clear GPU memory         │
│ ├─ Reload model             │
│ ├─ Record load time         │
│ ├─ Run 1 inference          │
│ └─ Record cold-start time   │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│ CLEANUP PHASE               │
│ ├─ Unload model             │
│ ├─ Stop monitoring          │
│ ├─ Free GPU memory          │
│ └─ Save results to JSONL    │
└─────────────────────────────┘
```

## Randomization & Reproducibility

**Fixed (for reproducibility):**
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Temperature: 0.1 (low, deterministic)
Top-p: 0.9 (standard)
Max tokens: 64 (per answer)
```

**Variable (between runs):**
- Order of models (randomized but logged)
- Exact CPU/GPU state (captured in logs)
- System load (recorded in metrics)

---

# DATA FLOW

## From Input to Output

```
SQuAD v2.0 Dataset (250 examples)
    ↓
┌─────────────────────────┐
│ For each model config   │
│  × 4 runtimes          │
│  × 3 quantizations     │
│  × 3 workloads         │
│  × 3-4 trials          │
└────────┬────────────────┘
         ↓
    EXPERIMENT RUNNER
    ├─ Load model
    ├─ For each example:
    │  ├─ Create prompt
    │  ├─ Run inference
    │  ├─ Collect metrics
    │  ├─ Evaluate task
    │  └─ Store row
    └─ Save results
         ↓
    experiment_results.jsonl
    (one line per trial)
         ↓
    RESULTS ANALYZER
    ├─ Load all results
    ├─ Aggregate by config
    ├─ Calculate statistics
    └─ Generate reports
         ↓
    ├── summary_report.txt (statistics)
    ├── decision_matrix.txt (recommendations)
    ├── results.csv (for Excel)
    └── benchmark_log.log (full trace)
```

---

# OUTPUT FORMAT

## Raw Results (experiment_results.jsonl)

Each line is a complete experiment result:

```json
{
  "config": {
    "model_name": "Qwen2.5-7B-Instruct",
    "runtime": "pytorch",
    "quantization": "fp16",
    "workload_type": "interactive_chat",
    "trial_number": 1,
    "is_cold_start": false
  },
  "status": "completed",
  "metrics": {
    "task_metrics": {
      "f1": 0.847,
      "exact_match": 0.75,
      "count": 20
    },
    "latency_stats_ms": {
      "min": 45.2,
      "max": 120.5,
      "mean": 78.3,
      "median": 75.0
    },
    "throughput_stats_tps": {
      "min": 40.5,
      "max": 65.3,
      "mean": 52.1,
      "median": 51.8
    },
    "memory_stats_mb": {
      "min": 13200,
      "max": 13850,
      "mean": 13500
    },
    "cpu_stats_percent": {
      "min": 5.2,
      "max": 15.3,
      "mean": 8.5
    },
    "samples_completed": 20
  }
}
```

## Aggregated Report (summary_report.txt)

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

TASK PERFORMANCE (SQuAD v2.0 F1 Score)
─────────────────────────────────────────────────────────────────
Model                           F1      Exact Match
Qwen2.5-7B                      0.847   0.72
Qwen2.5-1.5B                    0.812   0.68
Mistral-7B                      0.823   0.70
Phi-3.5-mini                    0.798   0.67
Llama-3.2-1B                    0.805   0.68

RUNTIME PERFORMANCE (Average TTFT, TPS)
─────────────────────────────────────────────────────────────────
Runtime          TTFT(ms)    TPS     Memory(MB)
llama.cpp        42.1        125.3   3200
pytorch          78.3        52.1    13850
onnx_runtime     65.2        58.3    13200
ollama           180.5       45.2    2400

QUANTIZATION IMPACT (Average F1, Memory)
─────────────────────────────────────────────────────────────────
Quantization    F1      Memory(MB)
FP16            0.845   13800
INT8            0.841   7000
Q4              0.832   3500
```

## Decision Matrix (decision_matrix.txt)

```
SCENARIO 1: Interactive Chat (Low Latency Critical)
Model: Qwen2.5-1.5B
Runtime: llama.cpp
Quantization: Q4
TTFT: 22ms | TPS: 155 | Memory: 650MB | F1: 0.81

SCENARIO 2: Quality First (High Accuracy)
Model: Qwen2.5-7B
Runtime: pytorch
Quantization: FP16
TTFT: 78ms | TPS: 52 | Memory: 14200MB | F1: 0.847

SCENARIO 3: Memory Constrained (Small Devices)
Model: Phi-3.5-mini
Runtime: llama.cpp
Quantization: Q4
TTFT: 28ms | TPS: 118 | Memory: 580MB | F1: 0.79

SCENARIO 4: Balanced (Good Performance + Efficiency)
Model: Phi-3.5-mini
Runtime: onnx_runtime
Quantization: INT8
TTFT: 35ms | TPS: 120 | Memory: 2600MB | F1: 0.79
```

---

## Summary

**What Gets Done:** Systematic benchmark of LLM inference on Windows
**Task:** SQuAD v2.0 extractive question answering
**Dataset:** 250 sampled examples from SQuAD v2.0 validation set
**Models:** 5 models (Qwen-1.5B, Qwen-7B, Llama-1B, Mistral-7B, Phi-3.5-mini)
**Metrics:** TTFT, TPS, Memory, F1, Exact Match, JSON Validity
**Output:** Decision matrix showing optimal config for each scenario

**Total Experiments:** ~200 configurations
**Total Time:** 8-12 hours
**Total Output:** One decision matrix with deployment recommendations
