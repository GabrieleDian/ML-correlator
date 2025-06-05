#!/bin/bash
#SBATCH --job-name=contr5L
#SBATCH --partition=allgpu
#SBATCH --chdir=/home/abekov/nonplanar_python
#SBATCH --array=1-50
#SBATCH --output=logs/slurm4D-%A_%a.out
#SBATCH --time=7-00:00:00        

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge

SLEEP_TIME=$((SLURM_ARRAY_TASK_ID))
sleep $SLEEP_TIME

nvidia-smi

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs on node $HOSTNAME"

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        (
        for n in $(seq 1 1 1000); do
                echo "Launching on GPU $GPU_ID"
                echo "Time now: $(date +"%T")"
                CUDA_VISIBLE_DEVICES=$GPU_ID python -u numbaContract5loopFF4d.py
        done
        ) &

        sleep 1
done

wait

