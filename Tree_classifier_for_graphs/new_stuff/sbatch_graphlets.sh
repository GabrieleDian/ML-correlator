#!/bin/bash
#SBATCH --job-name=motifs_job
#SBATCH --output=motifs_%A_%a.out
#SBATCH --error=motifs_%A_%a.err
#SBATCH --partition=allcpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --array=12
#SBATCH --hint=compute_bound
#SBATCH --distribution=block
#SBATCH --exclusive

# === ENVIRONMENT SETUP ===
unset LD_PRELOAD
source activate_module.sh
export PYTHONWARNINGS="ignore"


# === PATHS ===
cd /data/dust/user/gdian/ML-correlator/Tree_classifier_for_graphs/new_stuff

LOOP_ORDER=${SLURM_ARRAY_TASK_ID}
INPUT_FILE="../../Graph_Edge_Data/den_graph_data_${LOOP_ORDER}.csv"
OUTPUT_DIR="features/fgraphs"
OUTPUT_FILE="${OUTPUT_DIR}/${LOOP_ORDER}loops_motifs.parquet"

mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo " Job ID: ${SLURM_JOB_ID}  |  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo " Host: $(hostname)"
echo " Processing loop order: ${LOOP_ORDER}"
echo " Using ${SLURM_CPUS_PER_TASK} CPUs"
echo "========================================="

# === EXECUTION ===
python motif_features_cli.py \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --groups motifs34,motifs5,induced4,induced5 \
  --workers ${SLURM_CPUS_PER_TASK} \
  --intra-workers 2

echo "✓ Finished computing motifs for loop ${LOOP_ORDER}."
