#!/bin/bash
#SBATCH --job-name=lwf
#SBATCH --output=logs/TITAN/lwf_%j.out
#SBATCH --error=logs/TITAN/lwf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:a100:2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

REQUIRED_VRAM=20000
MAX_RETRIES=100000

mkdir -p logs

# =========================================================
# CHUAN BI MOI TRUONG
# =========================================================
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
source /datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/env/bin/activate

cd /datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/Benchmark-WSI/ || exit 1

echo "Before GPU selection:"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-EMPTY}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-EMPTY}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-EMPTY}"
echo "REQUIRED_VRAM=${REQUIRED_VRAM} MiB"

# =========================================================
# LOCAL GPU CHECKER
# Thay cho /usr/local/bin/gpu_check.sh bi loi nvidia-smi-i
# =========================================================
gpu_check_local() {
    local REQUIRED_VRAM=$1
    local JOB_ID=$2
    local MAX_RETRIES=$3

    local RESTART_COUNT
    RESTART_COUNT=$(scontrol show job "$JOB_ID" | grep -oP 'Restarts=\K\d+')

    if [ -z "$RESTART_COUNT" ]; then
        RESTART_COUNT=0
    fi

    echo "Restart count: $RESTART_COUNT" >&2

    if [ "$RESTART_COUNT" -ge "$MAX_RETRIES" ]; then
        echo "ERROR: Job $JOB_ID da requeue $RESTART_COUNT lan nhung van khong du ${REQUIRED_VRAM} MiB VRAM." >&2
        scancel "$JOB_ID"
        return 11
    fi

    # Giam kha nang nhieu job cung scan mot luc
    sleep $((RANDOM % 10))

    local BEST_GPU=""
    local BEST_FREE=0

    echo "Scanning GPUs..." >&2

    while IFS=',' read -r GPU NAME TOTAL USED UTIL; do
        GPU=$(echo "$GPU" | xargs)
        NAME=$(echo "$NAME" | xargs)
        TOTAL=$(echo "$TOTAL" | xargs)
        USED=$(echo "$USED" | xargs)
        UTIL=$(echo "$UTIL" | xargs)

        FREE=$((TOTAL - USED))

        echo "GPU $GPU | $NAME | used=${USED} MiB | free=${FREE} MiB | util=${UTIL}%" >&2

        # Chi chon card L40
        if [[ "$NAME" != *"L40"* && "$NAME" != *"A100"* ]]; then
            continue
        fi

        if [ "$FREE" -ge "$REQUIRED_VRAM" ] && [ "$FREE" -gt "$BEST_FREE" ]; then
            BEST_GPU="$GPU"
            BEST_FREE="$FREE"
        fi
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)

    if [ -z "$BEST_GPU" ]; then
        echo "WARNING: Khong co GPU nao du ${REQUIRED_VRAM} MiB VRAM. Dang requeue job $JOB_ID..." >&2
        scontrol requeue "$JOB_ID"
        return 10
    fi

    echo "Selected GPU $BEST_GPU with free VRAM ${BEST_FREE} MiB" >&2
    echo "$BEST_GPU"
    return 0
}

# =========================================================
# TU CHON GPU CO DU VRAM
# =========================================================
unset CUDA_VISIBLE_DEVICES

BEST_GPU=$(gpu_check_local "$REQUIRED_VRAM" "$SLURM_JOB_ID" "$MAX_RETRIES")
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "Job has been requeued because no GPU has enough VRAM."
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "Job failed after reaching max requeue retries."
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "gpu_check_local failed with exit code $EXIT_CODE"
    exit 1
fi

if ! [[ "$BEST_GPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: BEST_GPU is not a valid GPU index."
    echo "BEST_GPU=$BEST_GPU"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$BEST_GPU"

echo "=========================================="
echo "Selected physical GPU: $BEST_GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Required VRAM: ${REQUIRED_VRAM} MiB"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "=========================================="

nvidia-smi -i "$BEST_GPU"

# =========================================================
# PRIVATE MPS SERVER
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cleanup() {
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
trap cleanup EXIT

# =========================================================
# CUDA MEMORY CONFIG
# =========================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Dang xu ly"

python utils/main.py --config configs/methods.yaml --model lwf --backbone titan --folds all


echo "Hoan thanh!"

# Don dep MPS
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
