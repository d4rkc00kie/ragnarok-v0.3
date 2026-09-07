#!/usr/bin/env bash
# Detect CPU instruction sets relevant to Ollama / llama.cpp
set -euo pipefail

echo "=== CPU ==="
if command -v lscpu >/dev/null; then
  lscpu | grep -E 'Model name|Architecture|CPU\(s\)|Thread|Core|Socket|Flags|Vendor' || true
else
  grep -E 'model name|flags|cpu cores' /proc/cpuinfo | head -20
fi

echo
echo "=== SIMD / ML-relevant flags ==="
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
for f in sse sse2 ssse3 sse4_1 sse4_2 avx avx2 avx512f avx512bw avx512vl avx512vnni \
         avx_vnni amx_tile amx_bf16 amx_int8 fma f16c bmi2 movbe sha_ni; do
  if echo " $FLAGS " | grep -q " ${f} "; then
    echo "  [x] $f"
  else
    echo "  [ ] $f"
  fi
done

echo
echo "=== Physical cores (use for num_thread) ==="
if command -v lscpu >/dev/null; then
  PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  echo "  physical cores: $PHYS"
  echo "  logical CPUs:   $(nproc)"
else
  echo "  logical CPUs: $(nproc)"
fi

echo
echo "=== Guidance ==="
if echo " $FLAGS " | grep -q ' avx512f '; then
  echo "  AVX-512 present — llama.cpp can use wide SIMD (build must enable it)."
elif echo " $FLAGS " | grep -q ' avx2 '; then
  echo "  AVX2 present — standard fast path for modern Ollama CPU builds."
elif echo " $FLAGS " | grep -q ' avx '; then
  echo "  AVX only (no AVX2) — older path; prefer small Q4 models."
else
  echo "  No AVX — expect slow inference; use tiniest models only."
fi

if echo " $FLAGS " | grep -qi 'amd'; then
  :
fi
VENDOR=$(grep -m1 '^vendor_id' /proc/cpuinfo 2>/dev/null || true)
if echo "$VENDOR" | grep -qi AuthenticAMD; then
  echo "  AMD CPU: AVX2 is the main win; AVX-512 is rare on consumer Zen."
  echo "  Prefer Ollama stock binary; avoid forcing GPU offload to iGPU."
fi

echo
echo "=== Ollama process (if running) ==="
if command -v ollama >/dev/null; then
  ollama --version 2>/dev/null || true
  ollama ps 2>/dev/null || true
else
  echo "  ollama CLI not in PATH"
fi
