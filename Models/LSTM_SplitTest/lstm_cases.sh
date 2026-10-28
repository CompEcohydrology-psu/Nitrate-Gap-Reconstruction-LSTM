#!/bin/bash
#SBATCH --job-name=lstm_batch
#SBATCH --output=logs/lstm_batch_%j.out
#SBATCH --error=logs/lstm_batch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --gres=gpu:1

CONFIGS_PER_JOB="${CONFIGS_PER_JOB:-4}"
CASE_FILTER="${CASE_FILTER:-R20B14_* R20B60_* R20B365_*}"
SCENARIO_FILTER="${SCENARIO_FILTER:-}"
LSTM_EXTRA_ARGS="${LSTM_EXTRA_ARGS:-}"
SBATCH_TIME="${SBATCH_TIME:-}"
SBATCH_MEM="${SBATCH_MEM:-}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL="${LOCAL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

set -euo pipefail

MAX_PARALLEL_PER_GPU="${MAX_PARALLEL_PER_GPU:-${CONFIGS_PER_JOB}}"

_resolve_project_dir() {
  if [[ -n "${PROJECT_DIR:-}" && -d "${PROJECT_DIR}" ]]; then
    cd "${PROJECT_DIR}" && pwd
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    cd "${SLURM_SUBMIT_DIR}" && pwd
    return 0
  fi
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

PROJECT_DIR="$(_resolve_project_dir)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/lstm_cases.sh" ]]; then
  THIS_SCRIPT="${SLURM_SUBMIT_DIR}/lstm_cases.sh"
elif [[ -f "${PROJECT_DIR}/lstm_cases.sh" ]]; then
  THIS_SCRIPT="${PROJECT_DIR}/lstm_cases.sh"
else
  THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
fi
INPUTS_ROOT="${INPUTS_ROOT:-${PROJECT_DIR}/Inputs}"

if ! [[ "${CONFIGS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CONFIGS_PER_JOB must be a positive integer (got '${CONFIGS_PER_JOB}')" >&2
  exit 1
fi
if ! [[ "${MAX_PARALLEL_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_PARALLEL_PER_GPU must be a positive integer (got '${MAX_PARALLEL_PER_GPU}')" >&2
  exit 1
fi

run_worker() {
  [[ -n "${BATCH_FILE:-}" && -f "${BATCH_FILE}" ]] || {
    echo "ERROR: worker needs BATCH_FILE" >&2
    exit 1
  }

  cd "${PROJECT_DIR}"
  mkdir -p logs

  mapfile -t CONFIGS < <(grep -v '^[[:space:]]*$' "${BATCH_FILE}" || true)
  [[ ${#CONFIGS[@]} -gt 0 ]] || {
    echo "ERROR: empty BATCH_FILE: ${BATCH_FILE}" >&2
    exit 1
  }

  local JOB_ID="${SLURM_JOB_ID:-local}" FAIL=0
  local START_TIME END_TIME HOURS
  local -a PIDS=() LABELS=()
  local parallel="${MAX_PARALLEL_PER_GPU:-${#CONFIGS[@]}}"
  echo "Worker ${JOB_ID} on $(hostname) | ${#CONFIGS[@]} configs | parallel=${parallel}"
  [[ -n "${LSTM_EXTRA_ARGS}" ]] && echo "LSTM_EXTRA_ARGS=${LSTM_EXTRA_ARGS}"
  printf '  %s\n' "${CONFIGS[@]}"
  nvidia-smi || true
  START_TIME=$(date +%s)

  wait_batch() {
    [[ ${#PIDS[@]} -eq 0 ]] && return 0
    local i
    for i in "${!PIDS[@]}"; do
      if ! wait "${PIDS[$i]}"; then
        echo "FAILED ${LABELS[$i]}"
        FAIL=1
      else
        echo "Finished ${LABELS[$i]}"
      fi
    done
    PIDS=()
    LABELS=()
  }

  local cfg base case_name tag log_out log_err
  for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "${cfg}" ]]; then
      echo "FAILED missing: ${cfg}"
      FAIL=1
      continue
    fi
    base="$(basename "${cfg}" .yaml)"
    case_name="$(basename "$(dirname "${cfg}")")"
    tag="${case_name}__${base}"
    log_out="logs/${tag}__${JOB_ID}.out"
    log_err="logs/${tag}__${JOB_ID}.err"
    echo "Starting ${tag}"

    if [[ -n "${LSTM_EXTRA_ARGS}" ]]; then
      "${PYTHON_BIN}" main_LSTM.py --config "${cfg}" ${LSTM_EXTRA_ARGS} >"${log_out}" 2>"${log_err}" &
    else
      "${PYTHON_BIN}" main_LSTM.py --config "${cfg}" >"${log_out}" 2>"${log_err}" &
    fi
    PIDS+=("$!")
    LABELS+=("${tag}")
    if [[ ${#PIDS[@]} -ge "${parallel}" ]]; then
      wait_batch || true
    fi
  done
  wait_batch || true

  END_TIME=$(date +%s)
  HOURS=$(awk -v t=$((END_TIME - START_TIME)) 'BEGIN {printf "%.2f", t/3600}')
  echo "Batch done (fail=${FAIL}). ${HOURS} h"
  exit "${FAIL}"
}

resolve_case_dirs() {
  CASE_DIRS=()
  local -a requested=()
  if [[ "$#" -gt 0 ]]; then
    requested=("$@")
  elif [[ -n "${CASES:-}" ]]; then
    requested=(${CASES})
  fi

  if [[ ${#requested[@]} -gt 0 ]]; then
    local name dir matched cand
    for name in "${requested[@]}"; do
      dir="${INPUTS_ROOT}/${name}"
      if [[ -d "${dir}" ]]; then
        CASE_DIRS+=("${dir}")
        continue
      fi
      matched=0
      shopt -s nullglob
      for cand in "${INPUTS_ROOT}"/${name}*; do
        [[ -d "${cand}" ]] || continue
        CASE_DIRS+=("${cand}")
        matched=1
      done
      shopt -u nullglob
      if [[ "${matched}" -eq 0 ]]; then
        echo "ERROR: case not found: ${name}" >&2
        exit 1
      fi
    done
    return 0
  fi

  local dir pattern
  local -a patterns=(${CASE_FILTER})
  shopt -s nullglob
  for pattern in "${patterns[@]}"; do
    for dir in "${INPUTS_ROOT}"/${pattern}; do
      [[ -d "${dir}" ]] || continue
      CASE_DIRS+=("${dir}")
    done
  done
  shopt -u nullglob
}

collect_configs() {
  ALL_CONFIGS=()
  local case_dir cfg
  for case_dir in "${CASE_DIRS[@]}"; do
    while IFS= read -r cfg; do
      ALL_CONFIGS+=("${cfg}")
    done < <(find "${case_dir}" -maxdepth 1 -type f -name 'lstm_config_*.yaml' | sort)
  done
}

submit_batch() {
  local batch_idx="$1"
  shift
  local -a batch_cfgs=("$@")
  local batch_dir="${PROJECT_DIR}/logs/batches"
  mkdir -p "${batch_dir}"
  local batch_file job_name cfg
  batch_file="$(printf '%s/batch_%03d.txt' "${batch_dir}" "${batch_idx}")"
  : >"${batch_file}"
  for cfg in "${batch_cfgs[@]}"; do
    printf '%s\n' "${cfg}" >>"${batch_file}"
  done
  job_name="$(printf 'lstm_b%03d_n%d' "${batch_idx}" "${#batch_cfgs[@]}")"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${job_name} (${#batch_cfgs[@]} configs)"
    for cfg in "${batch_cfgs[@]}"; do
      echo "  $(basename "$(dirname "${cfg}")")/$(basename "${cfg}")"
    done
    return 0
  fi

  if [[ "${LOCAL}" == "1" ]]; then
    echo "LOCAL ${job_name}"
    BATCH_FILE="${batch_file}" \
      MAX_PARALLEL_PER_GPU="${MAX_PARALLEL_PER_GPU}" \
      PROJECT_DIR="${PROJECT_DIR}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      LSTM_EXTRA_ARGS="${LSTM_EXTRA_ARGS}" \
      bash "${THIS_SCRIPT}"
    return 0
  fi

  echo "Submitting ${job_name} (${#batch_cfgs[@]} configs, parallel=${MAX_PARALLEL_PER_GPU})"
  local -a sbatch_extra=()
  if [[ -n "${SBATCH_TIME}" ]]; then
    sbatch_extra+=(--time="${SBATCH_TIME}")
  fi
  if [[ -n "${SBATCH_MEM}" ]]; then
    sbatch_extra+=(--mem="${SBATCH_MEM}")
  fi
  sbatch \
    --job-name="${job_name}" \
    --chdir="${PROJECT_DIR}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},PYTHON_BIN=${PYTHON_BIN},BATCH_FILE=${batch_file},MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU}" \
    "${sbatch_extra[@]}" \
    "${THIS_SCRIPT}"
}

case_scenario_name() {
  local name="$1"
  if [[ "${name}" =~ _Seed[0-9]+_(.+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' "${name##*_}"
  fi
}

scenario_allowed() {
  local scen="$1"
  [[ -z "${SCENARIO_FILTER}" ]] && return 0
  local s
  local -a allowed=(${SCENARIO_FILTER})
  for s in "${allowed[@]}"; do
    [[ "${scen}" == "${s}" ]] && return 0
  done
  return 1
}

filter_case_dirs_by_scenario() {
  [[ -z "${SCENARIO_FILTER}" ]] && return 0
  local dir scen
  local -a kept=()
  for dir in "${CASE_DIRS[@]}"; do
    scen="$(case_scenario_name "$(basename "${dir}")")"
    if scenario_allowed "${scen}"; then
      kept+=("${dir}")
    fi
  done
  if [[ ${#kept[@]} -eq 0 ]]; then
    CASE_DIRS=()
  else
    CASE_DIRS=("${kept[@]}")
  fi
}

[[ -n "${BATCH_FILE:-}" ]] && run_worker

cd "${PROJECT_DIR}"
mkdir -p logs logs/batches
echo "Submit | PROJECT_DIR=${PROJECT_DIR} | INPUTS_ROOT=${INPUTS_ROOT}"

resolve_case_dirs "$@"
[[ ${#CASE_DIRS[@]} -gt 0 ]] || {
  echo "ERROR: no cases under ${INPUTS_ROOT} matching CASE_FILTER='${CASE_FILTER}'" >&2
  exit 1
}
declare -A _seen_case=()
_deduped=()
for dir in "${CASE_DIRS[@]}"; do
  [[ -n "${_seen_case[${dir}]+x}" ]] && continue
  _seen_case["${dir}"]=1
  _deduped+=("${dir}")
done
CASE_DIRS=("${_deduped[@]}")
unset _seen_case _deduped

filter_case_dirs_by_scenario
[[ ${#CASE_DIRS[@]} -gt 0 ]] || {
  echo "ERROR: no cases left after SCENARIO_FILTER='${SCENARIO_FILTER}'" >&2
  exit 1
}

collect_configs
[[ ${#ALL_CONFIGS[@]} -gt 0 ]] || {
  echo "ERROR: no lstm_config_*.yaml found" >&2
  exit 1
}

total="${#ALL_CONFIGS[@]}"
n_jobs=$(( (total + CONFIGS_PER_JOB - 1) / CONFIGS_PER_JOB ))
echo "CASE_FILTER='${CASE_FILTER}'"
echo "SCENARIO_FILTER='${SCENARIO_FILTER}'"
[[ -n "${LSTM_EXTRA_ARGS}" ]] && echo "LSTM_EXTRA_ARGS=${LSTM_EXTRA_ARGS}"
echo "Cases=${#CASE_DIRS[@]} Configs=${total} | CONFIGS_PER_JOB=${CONFIGS_PER_JOB} -> ${n_jobs} job(s) | parallel=${MAX_PARALLEL_PER_GPU}"
[[ -n "${SBATCH_TIME}" || -n "${SBATCH_MEM}" ]] && echo "SBATCH_TIME='${SBATCH_TIME}' SBATCH_MEM='${SBATCH_MEM}'"

SUBMITTED_JOBS=0
idx=0
batch_idx=0
while [[ "${idx}" -lt "${total}" ]]; do
  batch_idx=$((batch_idx + 1))
  batch=()
  local_j=0
  while [[ "${local_j}" -lt "${CONFIGS_PER_JOB}" && "${idx}" -lt "${total}" ]]; do
    batch+=("${ALL_CONFIGS[$idx]}")
    idx=$((idx + 1))
    local_j=$((local_j + 1))
  done
  submit_batch "${batch_idx}" "${batch[@]}"
  SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
done

echo "Done: ${SUBMITTED_JOBS} job(s) for ${total} config(s)."
[[ "${DRY_RUN}" == "1" ]] && echo "Re-run without DRY_RUN=1 to submit."
