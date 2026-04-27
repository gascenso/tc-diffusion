#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sweep_sampling_guidance.sh --run RUN_NAME [options]

Options:
  --run RUN_NAME       Trained diffusion run under runs/. Required unless RUN_NAME is set.
  --split SPLIT        Evaluation split for generated banks. Default: val.
  --n-per-class N      Samples per class per bank. Default: 100.
  --seed SEED          Generation and real-reference seed. Default: 123.
  --mode MODE          quick or expanded. Default: quick.
  --verbose 0|1        Forwarded to sample_bank/eval. Default: 1.

Environment overrides:
  RUN_NAME, SPLIT, N_PER_CLASS, SEED, MODE, VERBOSE, DC

Examples:
  scripts/sweep_sampling_guidance.sh --run pinn_loss_ramping_evalterms
  MODE=expanded N_PER_CLASS=100 scripts/sweep_sampling_guidance.sh --run pinn_loss_ramping_evalterms
EOF
}

RUN_NAME="${RUN_NAME:-}"
SPLIT="${SPLIT:-val}"
N_PER_CLASS="${N_PER_CLASS:-100}"
SEED="${SEED:-123}"
MODE="${MODE:-quick}"
VERBOSE="${VERBOSE:-1}"
DC="${DC:-./dc}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_NAME="${2:?--run requires a value}"
      shift 2
      ;;
    --split)
      SPLIT="${2:?--split requires a value}"
      shift 2
      ;;
    --n-per-class)
      N_PER_CLASS="${2:?--n-per-class requires a value}"
      shift 2
      ;;
    --seed)
      SEED="${2:?--seed requires a value}"
      shift 2
      ;;
    --mode)
      MODE="${2:?--mode requires a value}"
      shift 2
      ;;
    --verbose)
      VERBOSE="${2:?--verbose requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "--run RUN_NAME is required, or set RUN_NAME in the environment." >&2
  usage >&2
  exit 2
fi

if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
  echo "--split must be val or test, got: $SPLIT" >&2
  exit 2
fi

if [[ "$MODE" != "quick" && "$MODE" != "expanded" ]]; then
  echo "--mode must be quick or expanded, got: $MODE" >&2
  exit 2
fi

token_float() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  echo "$value"
}

bank_exists() {
  local bank="$1"
  [[ -f "outputs/${RUN_NAME}/sample_banks/${SPLIT}/${bank}/manifest.json" ]]
}

run_case() {
  local bank="$1"
  shift

  echo
  echo "==> ${bank}"

  if bank_exists "$bank"; then
    echo "[sweep] Sample bank already exists; skipping generation."
  else
    "$DC" sample_bank \
      --name "$RUN_NAME" \
      --split "$SPLIT" \
      --bank_name "$bank" \
      --n_per_class "$N_PER_CLASS" \
      --seed "$SEED" \
      --real_seed "$SEED" \
      --verbose "$VERBOSE" \
      --override "$@"
  fi

  "$DC" eval \
    --name "$RUN_NAME" \
    --split "$SPLIT" \
    --sample_bank "$bank" \
    --tag "$bank" \
    --generated_limit "$N_PER_CLASS" \
    --verbose "$VERBOSE"
}

run_baseline() {
  run_case "sg00_baseline_off_${SPLIT}_n${N_PER_CLASS}_seed${SEED}" \
    sampling_guidance.enabled=false
}

run_radial_step() {
  local step="$1"
  local step_token
  step_token="$(token_float "$step")"
  run_case "sg_radial_step${step_token}_start0p65_bw1_k64_${SPLIT}_n${N_PER_CLASS}_seed${SEED}" \
    sampling_guidance.enabled=true \
    sampling_guidance.target_split=train \
    sampling_guidance.step_size="$step" \
    sampling_guidance.guide_start_step_frac=0.65 \
    sampling_guidance.band_width_sigma=1.0 \
    sampling_guidance.neighbor_k=64 \
    sampling_guidance.radial_weight=1.0 \
    sampling_guidance.dav_weight=0.0 \
    sampling_guidance.hist_weight=0.0
}

run_radial_start() {
  local start="$1"
  local start_token
  start_token="$(token_float "$start")"
  run_case "sg_radial_step0p005_start${start_token}_bw1_k64_${SPLIT}_n${N_PER_CLASS}_seed${SEED}" \
    sampling_guidance.enabled=true \
    sampling_guidance.target_split=train \
    sampling_guidance.step_size=0.005 \
    sampling_guidance.guide_start_step_frac="$start" \
    sampling_guidance.band_width_sigma=1.0 \
    sampling_guidance.neighbor_k=64 \
    sampling_guidance.radial_weight=1.0 \
    sampling_guidance.dav_weight=0.0 \
    sampling_guidance.hist_weight=0.0
}

run_term_mix() {
  local label="$1"
  local radial="$2"
  local dav="$3"
  local hist="$4"
  run_case "sg_${label}_step0p005_start0p65_bw1_k64_${SPLIT}_n${N_PER_CLASS}_seed${SEED}" \
    sampling_guidance.enabled=true \
    sampling_guidance.target_split=train \
    sampling_guidance.step_size=0.005 \
    sampling_guidance.guide_start_step_frac=0.65 \
    sampling_guidance.band_width_sigma=1.0 \
    sampling_guidance.neighbor_k=64 \
    sampling_guidance.radial_weight="$radial" \
    sampling_guidance.dav_weight="$dav" \
    sampling_guidance.hist_weight="$hist"
}

echo "[sweep] run=${RUN_NAME} split=${SPLIT} n_per_class=${N_PER_CLASS} seed=${SEED} mode=${MODE}"
run_baseline

for step in 0.002 0.005 0.01 0.02; do
  run_radial_step "$step"
done

if [[ "$MODE" == "expanded" ]]; then
  for start in 0.4 0.75; do
    run_radial_start "$start"
  done

  run_term_mix "davonly_dav0p5" 0.0 0.5 0.0
  run_term_mix "histonly_hist0p5" 0.0 0.0 0.5
  run_term_mix "radial1_dav0p25" 1.0 0.25 0.0
  run_term_mix "radial1_hist0p25" 1.0 0.0 0.25
  run_term_mix "radial1_dav0p25_hist0p25" 1.0 0.25 0.25
fi

echo
echo "[sweep] Done. Metrics are under outputs/${RUN_NAME}/eval/${SPLIT}/<tag>/metrics.json."
