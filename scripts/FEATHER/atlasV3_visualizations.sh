#!/bin/bash
#SBATCH --job-name=atlasV3Vis
#SBATCH --output=logs/FEATHER/atlasV3Vis_%j.out
#SBATCH --error=logs/FEATHER/atlasV3Vis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:l40:2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -eo pipefail

REQUIRED_VRAM="${REQUIRED_VRAM:-10000}"
MAX_RETRIES="${MAX_RETRIES:-5}"
FOLD="${FOLD:-0}"
AFTER_TASK="${AFTER_TASK:-9}"
TSNE_TASKS="${TSNE_TASKS:-0,1,2,9}"
MAX_PER_CLASS="${MAX_PER_CLASS:-20}"
HEATMAP_TASK="${HEATMAP_TASK:-0}"
SLIDE_ID="${SLIDE_ID:-patient_125_node_0}"
THUMBNAIL="${THUMBNAIL:-/datastore/uittogether/LuuTru/Thuchd/Research/dataset/CAMELYON17/All/thumbnails/patient_125_node_0.jpg}"
WSI="${WSI:-}"
THUMBNAIL_MAX_SIZE="${THUMBNAIL_MAX_SIZE:-1600}"
REPO_ROOT=/datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/version_moi/Benchmark-WSI/
OUTPUT="${OUTPUT:-results/visualizations/atlas_v3/fold_${FOLD}_task_${AFTER_TASK}}"

mkdir -p "$REPO_ROOT/logs/FEATHER"
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
source /datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/env/bin/activate
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/huggingface_cache
export HF_HUB_DISABLE_XET=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-atlasv3-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/xdg-atlasv3-${SLURM_JOB_ID}"
unset PYTORCH_CUDA_ALLOC_CONF

gpu_check_local() {
    local required_vram=$1
    local job_id=$2
    local max_retries=$3
    local restart_count
    local best_gpu=""
    local best_free=0

    restart_count=$(scontrol show job "$job_id" | grep -oP 'Restarts=\K\d+' || true)
    restart_count=${restart_count:-0}
    if [ "$restart_count" -ge "$max_retries" ]; then
        echo "ERROR: Job $job_id da requeue $restart_count lan nhung van khong du ${required_vram} MiB VRAM." >&2
        scancel "$job_id"
        return 11
    fi

    sleep $((RANDOM % 10))
    while IFS=',' read -r gpu name total used util; do
        gpu=$(echo "$gpu" | xargs)
        name=$(echo "$name" | xargs)
        total=$(echo "$total" | xargs)
        used=$(echo "$used" | xargs)
        util=$(echo "$util" | xargs)
        local free=$((total - used))

        echo "GPU $gpu | $name | used=${used} MiB | free=${free} MiB | util=${util}%" >&2
        if [[ "$name" != *"L40"* && "$name" != *"A100"* ]]; then
            continue
        fi
        if [ "$free" -ge "$required_vram" ] && [ "$free" -gt "$best_free" ]; then
            best_gpu=$gpu
            best_free=$free
        fi
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)

    if [ -z "$best_gpu" ]; then
        echo "WARNING: Khong co GPU nao du ${required_vram} MiB VRAM; requeue job $job_id." >&2
        scontrol requeue "$job_id"
        return 10
    fi
    echo "Selected GPU $best_gpu with free VRAM ${best_free} MiB" >&2
    echo "$best_gpu"
}

unset CUDA_VISIBLE_DEVICES
set +e
BEST_GPU=$(gpu_check_local "$REQUIRED_VRAM" "$SLURM_JOB_ID" "$MAX_RETRIES")
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -eq 10 ]; then
    exit 0
elif [ "$EXIT_CODE" -ne 0 ]; then
    exit 1
fi
if ! [[ "$BEST_GPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid GPU index: $BEST_GPU" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cleanup() {
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
}
trap cleanup EXIT

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "FOLD=$FOLD AFTER_TASK=$AFTER_TASK TSNE_TASKS=$TSNE_TASKS"
echo "HEATMAP_TASK=$HEATMAP_TASK SLIDE_ID=$SLIDE_ID"
echo "WSI=${WSI:-<none>} THUMBNAIL=$THUMBNAIL"
nvidia-smi -i "$BEST_GPU"

VIS_ARGS=(
    --fold "$FOLD"
    --after-task "$AFTER_TASK"
    --tsne-tasks "$TSNE_TASKS"
    --max-per-class "$MAX_PER_CLASS"
    --heatmap-task "$HEATMAP_TASK"
    --slide-id "$SLIDE_ID"
    --thumbnail "$THUMBNAIL"
    --thumbnail-max-size "$THUMBNAIL_MAX_SIZE"
    --output "$OUTPUT"
)
if [ -n "$WSI" ]; then
    VIS_ARGS+=(--wsi "$WSI")
fi
python -u scripts/visualize_atlas_v3_methods.py "${VIS_ARGS[@]}"

echo "Hoan thanh ATLAS-v3 t-SNE va Grad-CAM visualizations: $OUTPUT"
