#!/bin/bash
#SBATCH --job-name=motifs_12loops
#SBATCH --output=motifs_12loops_%A_%a.out
#SBATCH --error=motifs_12loops_%A_%a.err
#SBATCH --partition=allcpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --hint=compute_bound
#SBATCH --exclusive

# === ENVIRONMENT ===
unset LD_PRELOAD
source activate_module.sh

echo "=== JOB CONTEXT ==="
hostname
pwd
echo "CPUs available: $SLURM_CPUS_PER_TASK"
which python
python -V
echo "===================="

# === DIRECTORIES ===
cd /data/dust/user/gdian/ML-correlator/Tree_classifier_for_graphs/new_stuff
mkdir -p features/fgraphs

# === ADAPTIVE SETTINGS ===
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-8}
WORKERS=$(( TOTAL_CPUS / 5 * 4 ))     # 80% for row-level
INTRA_WORKERS=$(( TOTAL_CPUS / WORKERS ))
if [ $INTRA_WORKERS -lt 1 ]; then INTRA_WORKERS=1; fi

echo "Detected ${TOTAL_CPUS} CPUs → using ${WORKERS} workers × ${INTRA_WORKERS} intra-workers"

# === START TOTAL TIMER ===
JOB_START=$(date +%s)

# === LOOP OVER 12-LOOP PARTS ===
for PART in $(seq 1 20); do
  INPUT_FILE="../../Graph_Edge_Data/den_graph_data_12_${PART}.csv"
  OUTPUT_FILE="features/fgraphs/12loops_part${PART}_spectral_plus_induced5.parquet"

  if [ ! -f "$INPUT_FILE" ]; then
    echo "⚠️  Missing input file: $INPUT_FILE — skipping"
    continue
  fi

  echo ">>> Processing part ${PART} (${INPUT_FILE})"
  START_TIME=$(date +%s)


  python motif_features_cli.py \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    --groups "spectral,induced5" \
    --workers ${WORKERS} \
    --intra-workers ${INTRA_WORKERS} \
    --ind4-max-samples 1000 \
    --no-progress

  END_TIME=$(date +%s)
  PART_MIN=$(( (END_TIME - START_TIME) / 60 ))
  echo "✓ Finished part ${PART} in ${PART_MIN} min"
done

# === END TOTAL TIMER ===
JOB_END=$(date +%s)
TOTAL_MIN=$(( (JOB_END - JOB_START) / 60 ))
TOTAL_HRS=$(awk "BEGIN {printf \"%.2f\", ${TOTAL_MIN}/60}")

echo "========================================"
echo "✓ All 12-loop motif parts processed"
echo "Total runtime: ${TOTAL_MIN} minutes (~${TOTAL_HRS} hours)"
echo "Completed on: $(date)"
echo "========================================"
