# IndicXlit Optimized Runtime (`indicxlit-serving`)

Hi! This repository contains my optimized setup for running **AI4Bharat's IndicXlit** transliteration model (typing in English letters like `namaste` and getting Hindi letters like `नमस्ते`).

The original model was slow, heavy, and failed the 100ms speed goal when typists typed longer words. I optimized it using **CTranslate2 INT8**, added a simple cache for repeated court words, and tested it thoroughly on an Intel CPU and integrated GPU. 

Now it runs in **under 15 ms**, uses **less than 35 MB of RAM**, and can run offline on a court clerk's laptop or serve 500+ users in the cloud for about **$156/month**.

> [!NOTE]
> **Full Engineering Report**: I wrote down all my experiments, hardware tests, cost math, and findings in simple detail on Notion:  
> **[Read the Full Report on Notion](https://app.notion.com/p/Adalat-AI-ML-Assignment-Report-3f361c2c3111800eb21fffd627288572)**

## Quick Numbers (TL;DR)

I tested this on an **Intel Core i5-1245U laptop** (10 cores, 12 threads) by sending 1,000 words across 12 concurrent workers with the cache turned off:

| What I Measured | Original Model (Fairseq FP32) | My Optimized Model (CTranslate2 INT8) | How Much Better |
| :--- | :--- | :--- | :--- |
| **Speed (Throughput)** | 44.3 requests/sec | **683.2 requests/sec** | **$15.4\times$ faster** |
| **Typical Latency (p50)** | 173 ms | **14 ms** | **$12.3\times$ faster** |
| **Slowest 5% Latency (p95)**| **362.9 ms (Too slow!)** | **40.6 ms (Passed <100ms goal)** | **$8.9\times$ faster** |
| **Model Size on Disk** | 132 MB | **12.8 MB** | **$10\times$ smaller** |
| **RAM Usage** | ~800 MB per worker | **~35 MB** | **$22\times$ less RAM** |
| **Startup / Load Time** | 4.5 seconds | **0.038 seconds (38 ms)** | **$118\times$ faster** |
| **Accuracy (Dakshina test)**| 59.6% (sample) | **61.0% (all 4,442 words)** | **No loss in accuracy (0% regression)** |
| **Cloud Cost (500 users)** | ~$1,861 / month | **~$156 / month** | **91% cheaper** |

## Setup & Installation

I recommend using **Python 3.10**.

```bash
# 1. Create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install requirements (CPU only, no heavy CUDA download)
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> The converted 12.8 MB INT8 model is already in `models/ct2_indicxlit_int8/`, so you don't have to download or convert any big model files.

## How to Run the Benchmarks

### 1. Run the Main Comparison Benchmark (`bench_suite.py`)
This script runs the original model and my optimized model side-by-side using 12 threads:

```bash
# Runs 1,000 words across 12 threads with no cache (pure neural speed)
python benchmarks/bench_suite.py -n 1000 -c 12 --no-cache --backends all
```

Options you can change:
- `-n`: How many words to test (e.g. `1000`).
- `-c`: How many threads to use (e.g. `1`, `4`, `12`).
- `--no-cache`: Turns off the cache so you only test the model itself.
- `--backends`: Can be `all`, `ct2`, or `stock`.

### 2. Run the Intel OpenVINO Hardware Test (`bench_openvino.py`)
I also tested running the model with OpenVINO on the CPU vs the Intel Iris Xe integrated GPU:

```bash
# Test FP32 precision
python benchmarks/bench_openvino.py -p fp32

# Test INT8 precision
python benchmarks/bench_openvino.py -p int8
```

### 3. Check Quality & Accuracy (`eval_quality.py`)
To prove that INT8 quantization did not hurt transliteration accuracy, run the evaluation on the full **Google Dakshina Hindi test set** (4,442 words):

```bash
python eval_quality.py -n 0
```

Results:
- **Top-1 Accuracy**: **61.01%** (slightly higher than the 60.56% reported in the original AI4Bharat paper).
- **Top-4 Accuracy**: **85.23%** (meaning the right Hindi word is almost always in the top 4 suggestions).
- **Regression**: **0.00%** (exact same predictions as the original model).

## Running the API Server

I created a simple FastAPI server so typists or web editors can get suggestions over HTTP.

> [!NOTE]
> **Honest note on the API**: I didn't spend much time building a complicated API. Most of my time went into the **benchmarks, testing different ideas, and making the model fast**. The FastAPI server here is simple, clean, and gets the job done to show real latency and Docker support.

### Run Locally with Uvicorn:
```bash
# Start the server on port 8000
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

In another terminal, test it with my test script:
```bash
python test_server.py
```

Or test with `curl`:
```bash
# Health check
curl http://localhost:8000/health

# Transliterate one word
curl -X POST http://localhost:8000/transliterate \
  -H "Content-Type: application/json" \
  -d '{"word": "adhivakta", "topk": 4}'

# Transliterate a batch of words
curl -X POST http://localhost:8000/transliterate_batch \
  -H "Content-Type: application/json" \
  -d '{"words": ["vakeel", "nyayalaya", "faisla", "mukadma"], "topk": 4}'

# Check latency stats and cache hit rate
curl http://localhost:8000/stats
```

### Run with Docker:
The container image is small (~337 MB) because it doesn't include PyTorch or CUDA:

```bash
# Build the container
docker build -t xlit-server .

# Run it
docker run -d -p 8000:8000 --name xlit-service xlit-server

# Test health
curl http://localhost:8000/health
```

## What Each File Does

Here is a quick map of the files in this repo:

```text
indicxlit-serving/
├── README.md                      # This guide
├── Dockerfile                     # Small Docker container recipe
├── requirements.txt               # Dependencies (CPU only)
├── engine.py                      # Core engine with CTranslate2 INT8 and cache
├── eval_quality.py                # Tests accuracy on 4,442 Google Dakshina words
├── test_server.py                 # Quick test to make sure all API endpoints work
│
├── benchmarks/
│   ├── bench_suite.py             # Main benchmark comparing original vs optimized
│   ├── bench_openvino.py          # Benchmark comparing Intel CPU vs Iris Xe GPU
│   └── workloads.py               # Test words (court words, short/medium/long words)
│
├── server/
│   ├── app.py                     # FastAPI routes (/health, /transliterate, /stats)
│   ├── config.py                  # Settings (threads, model path, cache size)
│   ├── schemas.py                 # Request and response formats
│   ├── metrics.py                 # Tracks latency and cache hits
│   └── requirements.txt           # Minimal requirements for the Docker container
│
├── models/
│   └── ct2_indicxlit_int8/        # The 12.8 MB converted INT8 model files
│
└── results/                       # Saved JSON files with benchmark numbers from my machine
    ├── bench_n1000_c12_mixed_latest.json
    ├── intel_openvino_latest.json
    └── dakshina_eval.json
```

## Where I Spent My Time & What I'd Improve Next

As I mentioned, **I spent most of my time on the model speed and benchmarking side**:
1. Finding out why the original model was slow (the Python loop and thread contention).
2. Trying different ideas:
   - `torch.compile` (didn't work on this model).
   - ONNX Runtime (slow because Python and C++ had to talk for every single letter).
   - Intel GPU with OpenVINO (great for batches at 41,000 words/sec, but CPU was still faster for single words).
   - CTranslate2 INT8 (the best choice: fast, light, and low latency).
3. Fixing a thread bug in Docker: with 4 internal threads, latency jumped to 670ms because threads were fighting over tiny matrices. Setting internal threads to 1 fixed it and dropped latency back down to 12ms.
4. Testing accuracy on Google Dakshina to ensure 0% regression.

Because I focused on the model and benchmarks, the API server is kept basic. Here is what I would improve on the API side given more time:
- **Write the server in C++ or Rust**: Instead of wrapping CTranslate2 in Python, a native C++ or Rust server would be even faster.
- **Use local Unix sockets**: If running directly on a clerk's laptop, using a Unix socket instead of HTTP would save a few milliseconds.
- **Add WebSockets**: Suggesting words character-by-character as the clerk types would feel smoother than sending requests word-by-word.
- **Dynamic batching**: If hundreds of people type at the exact same millisecond, grouping requests together would make the server handle even higher traffic easily.

*For the full cost breakdown and all details, check out my [Full Report on Notion](https://app.notion.com/p/Adalat-AI-ML-Assignment-Report-3f361c2c3111800eb21fffd627288572).*
