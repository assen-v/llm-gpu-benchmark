# LLM GPU Benchmark

Benchmarking a small open LLM (Qwen2.5-1.5B-Instruct) on an NVIDIA Tesla T4 GPU: speed, memory, and answer quality across numeric precisions (fp32, fp16, int8, int4) and batch sizes, plus CUDA kernels written with Numba and compared against cuBLAS.

Everything here ran on a free Google Colab T4. All numbers come from the notebook in this repo, and the limitations section lists what they do and don't show.

## Summary of findings

- **GPU vs CPU:** the same model generating 32 tokens ran about 15x faster on the T4 (fp32, batch 1) than on a single CPU thread. A later session with an unrecorded CPU thread count measured only about 4x, so treat the CPU baseline as weak.
- **Batching is the main speed lever.** In fp16, going from batch 1 to batch 16 raised throughput from 20.7 to 339.3 tokens/s (16.4x) while the time per batch stayed around 1.5 s. A profile at batch 1 showed matrix-vector kernels using about 71% of GPU time and the CPU busy about 3x longer than the GPU, which is consistent with the GPU being underused at batch 1. I did not test that explanation directly.
- **fp16 was the best default.** It matched fp32 on accuracy (32 vs 31 of 50 correct), used half the memory (3.18 vs 6.29 GB at batch 16), and was 1.4x faster at batch 16.
- **int4 had the smallest footprint** (1.30 GB, about 40% of fp16) but ran at about half fp16's throughput and scored lower on GSM8K (23 vs 32 of 50).
- **bitsandbytes int8 was the slowest** (about 10x slower than fp16 in the speed test, 5x slower on the eval) and also scored lower (24 of 50).
- **Hand-written CUDA matmul kernels were about 26 to 28x slower than cuBLAS** on a 1024x1024 float32 multiply. Shared-memory tiling gave only about a 4 to 5% gain over the naive kernel.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla T4 (15 GB), Google Colab free tier |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Python / PyTorch | Python 3.13, PyTorch 2.11.0+cu130 |
| Libraries | transformers, accelerate, bitsandbytes, datasets, numba — **TODO: add versions** (`!pip list \| grep -E "transformers\|accelerate\|bitsandbytes\|datasets\|numba"`) |

The T4 has no bf16 support, so fp16 is the half-precision format used.

## Method

**Speed benchmark** (`bench` function in the notebook)
- 16 fixed short prompts, reused for every test. Batch size 1 uses the first prompt, batch 4 the first four, and batch 16 all of them.
- Greedy decoding, exactly 32 new tokens per prompt (`min_new_tokens == max_new_tokens`), so every run does the same work.
- GPU timing uses CUDA events with `torch.cuda.synchronize()`, because GPU work is asynchronous and a plain timer would only measure the launch.
- A warmup generation runs first and is discarded. Each reported number is the mean of 3 timed runs (the CPU baseline used 2 runs).
- Throughput is `32 x batch size / latency`. Memory is `torch.cuda.max_memory_allocated()` (tensor memory, which reads lower than `nvidia-smi`).

**Accuracy evaluation**
- First 50 questions of the GSM8K test set, prompted in the Qwen chat format with "Solve step by step. End with 'Answer: <number>'."
- Greedy decoding, up to 512 new tokens, batch size 8.
- Strict scoring: a reply counts only if it contains `Answer:` and the first number after it equals the reference. Replies without `Answer:` are counted as wrong and reported separately as "unfinished".
- All four precisions use the same questions, prompt, token budget, and scoring.

**Quantization:** int8 and int4 use bitsandbytes (`load_in_8bit`; `load_in_4bit` with fp16 compute dtype).

## Results

### Speed and memory

![Throughput and memory](<precision_charts (1).png>)

| Precision | tok/s, batch 1 | tok/s, batch 4 | tok/s, batch 16 | Peak memory, batch 16 (GB) | Latency per batch, batch 16 (s) |
|---|---|---|---|---|---|
| fp32 | 22.6 | 86.8 | 242.8 | 6.29 | 2.11 |
| fp16 | 20.7 | 87.9 | 339.3 | 3.18 | 1.51 |
| int8 | 2.2 | 8.5 | 34.1 | 1.87 | 15.02 |
| int4 | 11.9 | 42.8 | 170.3 | 1.30 | 3.01 |

Full data: `precision_results.csv`. Batch-1 numbers vary by roughly 20 to 25% between runs, so small differences at batch 1 (for example fp32 vs fp16) are noise.

### Accuracy (GSM8K, first 50 questions, 512 new tokens)

| Precision | Correct (of 50) | Unfinished | Generation time | Paired vs fp16 (lost / gained) |
|---|---|---|---|---|
| fp32 | 31 | 8 | 198 s | not computed |
| fp16 | 32 | 8 | 107 s | n/a |
| int8 | 24 | 10 | 536 s | lost 10 / gained 2 |
| int4 | 23 | 5 | 229 s | lost 16 / gained 7 |

"Lost" means fp16 got the question right and the quantized model got it wrong; "gained" is the reverse. Raw replies: `gsm8k_results.json`.

An early run at 256 new tokens left 25 of 50 fp16 replies unfinished, so the token budget was raised to 512 before comparing precisions. At 512, 8 fp16 replies were still unfinished.

**What the wrong answers looked like.** I read three wrong answers each for int8 and int4. Five of the six were misreadings of a multi-step setup (for example applying a percentage to the wrong base, or misreading a "every second glass is discounted" rule), and one was truncated. None looked like a plain arithmetic slip. Six examples is anecdotal, not a measurement.

### Profile at batch 1 (fp16, 16 new tokens)

| | Time |
|---|---|
| Self CPU time | about 858 ms |
| Self CUDA time | about 276 ms |
| `gemv` kernels (matrix times one vector) | about 71% of GPU time |

The profiler adds CPU-side overhead, so the 3x ratio is approximate.

## CUDA kernels (Numba)

All kernels are in the notebook and checked against a reference result.

| Kernel | Result |
|---|---|
| Vector add, 10M float32 | matches NumPy (no timing taken) |
| Naive matmul, 1024x1024 float32 | about 22.9 ms (about 94 GFLOP/s), about 28x slower than cuBLAS |
| Tiled matmul (16x16 shared memory) | about 21.9 ms (about 98 GFLOP/s), about 26x slower than cuBLAS |
| cuBLAS via `torch.matmul` | about 0.83 ms (about 2,590 GFLOP/s) |

Times are medians of repeated runs after warmup. My first single measurements were misleading: naive 34.7 ms and cuBLAS 0.40 ms, a gap of about 86x. Re-timing with warmup and repeats shrank it to 26 to 28x, and cuBLAS itself varied from 0.40 to 0.83 ms between runs, probably from GPU clock and power management (not verified).

Tiling beat the naive kernel by only about 4 to 5% consistently across runs. cuBLAS is much faster because of optimizations I did not implement (for example each thread computing several outputs from registers, larger tiles, and vectorized loads), so this is a measured gap, not a tuned comparison.

## Limitations

- **One GPU, one model, free-tier Colab.** The T4 is shared and its performance varies between runs. Results come from several sessions (Colab sessions reset), so absolute numbers are not strictly comparable across sections.
- **Small evaluation set.** With 50 questions, one question is 2 percentage points, and each score has roughly a plus or minus 14 point interval. An exact two-sided sign test on the paired disagreements gives p of about 0.04 for int8 (10 lost vs 2 gained) and about 0.09 for int4 (16 vs 7). So int8's drop is probably real, and int4's is suggestive but not clear. A larger set would be needed to quantify the cost.
- **Single greedy run per setting,** with no variance estimate.
- **Unfinished replies** (5 to 10 per precision) count as wrong, so part of each score reflects the token budget and not the math.
- **bitsandbytes only.** int8 and int4 results are specific to that library and this model size on a T4. Other quantization methods may behave differently. bitsandbytes also logged a bf16-to-fp16 cast warning on every layer in int8, which I did not investigate.
- **Weak CPU baseline:** the first CPU run used 1 thread (1.75 tok/s). A later session measured 5.58 tok/s with an unrecorded thread count.
- **Explanations are hypotheses.** Statements about why things are slow (launch overhead at batch 1, unpacking cost for quantized weights, memory traffic in the naive kernel) are consistent with the measurements but were not tested directly.

## What I'd test next

- A larger GSM8K sample (200+ questions), to tighten the accuracy comparison.
- Serving with vLLM or Triton and measuring throughput under concurrent requests.
- Other quantization methods (GPTQ, AWQ) and the dtype used in the bitsandbytes int8 path.
- Longer prompts and outputs, where memory and KV-cache costs grow with batch size.
- A register-blocked matmul kernel, to see how much of the cuBLAS gap it closes.

## Files

| File | What it is |
|---|---|
| `01_baseline.ipynb` | The full notebook, in the order it was run (setup, benchmark, profiling, precision sweep, CUDA kernels, GSM8K evaluation) |
| `precision_results.csv` | Speed and memory for each precision and batch size |
| `precision_charts (1).png` | Throughput vs batch size, and peak memory by precision |
| `gsm8k_results.json` | Per-question predictions, references, and raw replies for each precision |

## Reproduce

1. Open `01_baseline.ipynb` in Google Colab and select a **T4 GPU** runtime.
2. Run the install, imports, `bench`, `prompts`, and `load` cells first, then the section you want. The GSM8K evaluation needs the dataset, `make_prompt`, `run_eval`, and the strict `extract` cells.
3. Expect the precision sweep to take 10 to 20 minutes and the GSM8K evaluation about 20 minutes (int8 is the slowest). Keep the machine plugged in, since Colab sessions can disconnect.
