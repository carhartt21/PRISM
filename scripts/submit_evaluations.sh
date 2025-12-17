#!/usr/bin/env bash
set -euo pipefail

# submit_evaluations.sh
# Submit evaluation runs to LSF for a list of model names using scripts/run_evaluation.
#
# Usage:
#   ./scripts/submit_evaluations.sh MODEL1 MODEL2 ... -- [extra args forwarded to run_evaluation]
# Example:
#   ./scripts/submit_evaluations.sh CUT MPRNet -- --regenerate-manifest
#
# Environment/customization:
#   - QUEUE: LSF queue (default BatchGPU)
#   - GPUSPEC: GPU resource string (default: num=1:mode=exclusive_process:gmem=24G)
#   - You can also set other LSF options inside the script if needed.

QUEUE="${QUEUE:-BatchGPU}"
GPUSPEC="${GPUSPEC:-num=1:mode=exclusive_process:gmem=24G}"

# Collect models until `--` then collect extra args
models=()
extra_args=()

while [[ $# -gt 0 ]]; do
  if [[ $1 == '--' ]]; then
    shift
    extra_args=("${@}")
    break
  fi
  models+=("$1")
  shift
done

# Default models if none provided (edit as needed)
if [ ${#models[@]} -eq 0 ]; then
  models=(CUT)
  echo "No models provided; defaulting to: ${models[*]}"
fi

# LSF base options (kept in array for safe word-splitting)
BSUB_BASE=( -gpu "${GPUSPEC}" -q "${QUEUE}" -R "span[hosts=1]" -n 6 -L /bin/bash )

for model in "${models[@]}"; do
  # Sanitize jobname (only alnum + underscore)
  jobname="eval_${model//[^a-zA-Z0-9_]/_}"
  out_log="logs/lsf_${jobname}_%J_gpu.log"
  err_log="logs/lsf_${jobname}_%J_gpu.err"

  # Build command to run: call the run_evaluation script with model as arg
  # Special handling for --regenerate-manifest which must be first arg
  cmd_parts=()
  if [[ " ${extra_args[*]} " =~ " --regenerate-manifest " ]]; then
    cmd_parts+=("--regenerate-manifest" "$model")
  else
    cmd_parts+=("$model")
  fi
  
  # Add remaining extra args
  for arg in "${extra_args[@]:-}"; do
    if [[ "$arg" != "--regenerate-manifest" ]]; then
      cmd_parts+=("$arg")
    fi
  done

  # Build final command string with proper quoting
  cmd="bash scripts/run_evaluation"
  for part in "${cmd_parts[@]}"; do
    cmd+=" $(printf '%q' "$part")"
  done

  echo "Submitting job for model='${model}' -> jobname='${jobname}'"
  echo "  bsub -gpu \"${GPUSPEC}\" -q ${QUEUE} -R \"span[hosts=1]\" -n 6 -oo \"${out_log}\" -eo \"${err_log}\" -L /bin/bash -J \"${jobname}\" \"${cmd}\""

  # Submit job
  # Note: We expand BSUB_BASE in a way that bsub receives the options properly.
  bsub -gpu "${GPUSPEC}" \
       -q "${QUEUE}" \
       -R "span[hosts=1]" \
       -n 6 \
       -oo "${out_log}" \
       -eo "${err_log}" \
       -L /bin/bash \
       -J "${jobname}" \
       "${cmd}"

  sleep 0.2
done

echo "Submitted ${#models[@]} jobs."
