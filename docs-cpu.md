# Ollama CPU performance (AMD iGPU / no discrete GPU)

## 1. Model choice (biggest win)

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull phi3:mini
# quality step-up (still CPU-friendly if RAM ≥ 16 GB):
ollama pull qwen2.5:7b-instruct-q4_K_M
```

Prefer **Q4_K_M**. Avoid 14B+ on CPU unless you accept multi-minute replies.

## 2. Threads = physical cores

```bash
# physical cores (Fedora)
lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l
```

In Silent Precision:

```
/speed amd
/params set num_thread <physical_cores>
/params set num_gpu 0
```

Do **not** set threads to 2× cores (hyperthreads); that often slows token generation.

## 3. Ollama server environment

Create/override systemd user or system service, or export before `ollama serve`:

```bash
export OLLAMA_KEEP_ALIVE=60m
export OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
# force CPU if something tries GPU offload:
export CUDA_VISIBLE_DEVICES=-1
# AMD: leave ROCm alone unless you know it helps your APU
```

## 4. Runtime options (API / TUI)

| Option | CPU recommendation |
|--------|-------------------|
| `num_gpu` | `0` |
| `num_thread` | physical cores |
| `num_ctx` | 1024–2048 |
| `num_predict` | 128–256 |
| `num_batch` | 32–128 |
| `use_mmap` | true |
| `use_mlock` | false (unless you have spare RAM) |

## 5. OS

```bash
# performance governor (laptop AC power)
sudo cpupower frequency-set -g performance

# less background noise
# close browsers / Electron apps while generating
```

## 6. Silent Precision quick path

```
/model qwen2.5:3b
/speed amd
/params set num_thread 6
/tools off
```

## 7. Measure

```bash
time ollama run qwen2.5:3b "Count from 1 to 20."
ollama ps
```

Watch tokens/s; tune `num_thread` up/down by 1–2 and retest.
