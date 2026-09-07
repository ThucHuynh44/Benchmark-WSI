#!/bin/bash
#SBATCH --job-name=install_gigapath
#SBATCH --output=logs/GIGAPATH/install_%j.out
#SBATCH --error=logs/GIGAPATH/install_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:l40:2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -eo pipefail

mkdir -p logs/GIGAPATH

module clear -f
module load slurm/slurm/24.11

source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
conda activate benchmark_gigapath

cd /datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/Benchmark-WSI

export PYTHONUNBUFFERED=1

echo "========================================"
echo "JOB INFO"
echo "========================================"
echo "JOB ID   = $SLURM_JOB_ID"
echo "HOST     = $(hostname)"
echo "DATE     = $(date)"
echo "ENV      = $CONDA_PREFIX"
echo "PYTHON   = $(which python)"
echo

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-EMPTY}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-EMPTY}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-EMPTY}"
echo

# ============================================================
# GPU INFORMATION
# ============================================================

echo "========================================"
echo "GPU FROM SLURM"
echo "========================================"

nvidia-smi -L

echo

# KHONG DOI CUDA_VISIBLE_DEVICES O DAY

# ============================================================
# TEST TORCH
# ============================================================

echo "========================================"
echo "TEST TORCH"
echo "========================================"

python -u <<'PY'
print("1. importing torch...", flush=True)

import torch
import numpy as np

print("2. import torch OK", flush=True)
print("numpy             :", np.__version__, flush=True)
print("torch             :", torch.__version__, flush=True)
print("torch.version.cuda:", torch.version.cuda, flush=True)

assert torch.__version__.startswith("2.0.0")
assert torch.version.cuda == "11.8"

print("3. calling torch.cuda.is_available()...", flush=True)

available = torch.cuda.is_available()

print("4. torch.cuda.is_available() returned", flush=True)
print("CUDA available    :", available, flush=True)

if not available:
    raise RuntimeError("CUDA is not available in this SLURM job")

print("GPU count         :", torch.cuda.device_count(), flush=True)
print("GPU logical id    : 0", flush=True)
print("GPU name          :", torch.cuda.get_device_name(0), flush=True)
print("Capability        :", torch.cuda.get_device_capability(0), flush=True)

major, minor = torch.cuda.get_device_capability(0)

assert (major, minor) == (8, 9), \
    f"Expected NVIDIA L40 SM 8.9, got {major}.{minor}"

print("5. CUDA initialization OK", flush=True)
PY

# ============================================================
# CHECK CURRENT NVCC
# ============================================================

echo
echo "========================================"
echo "CURRENT NVCC"
echo "========================================"

which nvcc || true
nvcc --version || true

# ============================================================
# INSTALL CUDA 11.8 COMPILER IF NECESSARY
# ============================================================

NVCC118=0

if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    if "$CONDA_PREFIX/bin/nvcc" --version 2>/dev/null | grep -q "release 11.8"; then
        NVCC118=1
    fi
fi

if [ "$NVCC118" -eq 0 ]; then

    echo
    echo "========================================"
    echo "INSTALL CUDA TOOLKIT 11.8"
    echo "========================================"

    conda install -y \
        -c nvidia/label/cuda-11.8.0 \
        cuda-toolkit=11.8.0
fi

# ============================================================
# CUDA HOME
# ============================================================

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

echo
echo "========================================"
echo "CUDA COMPILER"
echo "========================================"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc=$(which nvcc)"

nvcc --version

if ! nvcc --version 2>/dev/null | grep -q "release 11.8"; then
    echo "ERROR: nvcc is not CUDA 11.8"
    exit 1
fi

echo "CUDA compiler 11.8 OK"

# ============================================================
# BUILD DEPENDENCIES
# ============================================================

echo
echo "========================================"
echo "BUILD DEPENDENCIES"
echo "========================================"

python -m pip install \
    "setuptools<81" \
    wheel \
    "packaging==23.2" \
    "ninja==1.11.1.1"

which ninja
ninja --version

# 2 CPUs allocated
export MAX_JOBS=2

# NVIDIA L40 = Ada Lovelace = SM89
export TORCH_CUDA_ARCH_LIST="8.9"

echo "MAX_JOBS=$MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ============================================================
# INSTALL FLASH ATTENTION
# ============================================================

echo
echo "========================================"
echo "INSTALL FLASH-ATTN 2.5.8"
echo "========================================"

python -m pip install \
    "flash-attn==2.5.8" \
    --no-build-isolation

# ============================================================
# IMPORT TEST
# ============================================================

echo
echo "========================================"
echo "FLASH-ATTN IMPORT TEST"
echo "========================================"

python -u <<'PY'
print("Import flash_attn...", flush=True)

import torch
import flash_attn
from flash_attn import flash_attn_func

print("flash-attn        :", flash_attn.__version__, flush=True)
print("torch             :", torch.__version__, flush=True)
print("torch CUDA        :", torch.version.cuda, flush=True)
print("CUDA available    :", torch.cuda.is_available(), flush=True)
print("GPU               :", torch.cuda.get_device_name(0), flush=True)
print("Capability        :", torch.cuda.get_device_capability(0), flush=True)

print("FlashAttention import OK", flush=True)
PY

# ============================================================
# FLASH ATTENTION REAL CUDA TEST
# ============================================================

echo
echo "========================================"
echo "FLASH-ATTN CUDA TEST"
echo "========================================"

python -u <<'PY'
import torch
from flash_attn import flash_attn_func

print("Creating CUDA tensors...", flush=True)

q = torch.randn(
    1, 128, 8, 64,
    device="cuda",
    dtype=torch.float16,
    requires_grad=True
)

k = torch.randn_like(q)
v = torch.randn_like(q)

print("Calling flash_attn_func...", flush=True)

out = flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    causal=False
)

torch.cuda.synchronize()

print("Output:", out.shape, flush=True)

loss = out.float().mean()
loss.backward()

torch.cuda.synchronize()

print("FlashAttention forward  : OK", flush=True)
print("FlashAttention backward : OK", flush=True)
PY

# ============================================================
# TEST GIGAPATH
# ============================================================

echo
echo "========================================"
echo "TEST GIGAPATH"
echo "========================================"

python -u <<'PY'
import torch
import flash_attn
import gigapath

print("torch      :", torch.__version__, flush=True)
print("torch CUDA :", torch.version.cuda, flush=True)
print("flash-attn :", flash_attn.__version__, flush=True)
print("GPU        :", torch.cuda.get_device_name(0), flush=True)
print("gigapath   : import OK", flush=True)
PY

echo
echo "========================================"
echo "BENCHMARK_GIGAPATH READY"
echo "========================================"
echo "Finished: $(date)"
