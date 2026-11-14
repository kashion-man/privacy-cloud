#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command_name="${1:-help}"
if (( $# )); then shift; fi
python_bin="${PRIVACY_PYTHON:-python3}"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

case "$command_name" in
  agent)    exec "$python_bin" -m privacy_infer.agent_cli "$@" ;;
  infer)    exec "$python_bin" -m privacy_infer "$@" ;;
  he-only)  export QWEN_HE_PYTHON="${QWEN_HE_PYTHON:-$python_bin}"; exec bash "$root/he_only/run.sh" "$@" ;;
  he-gpu)   exec bash "$root/he_only/run_perseus.sh" "$@" ;;
  he-multi) exec bash "$root/he_only/run_perseus_multi.sh" "$@" ;;
  calibrate) exec "$python_bin" "$root/he_only/calibrate.py" "$@" ;;
  he-mpc)   exec "${HE_MPC_PYTHON:-$python_bin}" "$root/he_mpc/client.py" "$@" ;;
  he-mpc-server) exec "${HE_MPC_PYTHON:-$python_bin}" "$root/he_mpc/server.py" "$@" ;;
  fetch-data) exec "$python_bin" "$root/scripts/fetch_privacy_fed_sources.py" "$@" ;;
  build-data) exec "$python_bin" "$root/scripts/build_seed_dataset.py" "$@" ;;
  revise-data) exec "$python_bin" "$root/scripts/build_training_dataset.py" "$@" ;;
  pack-train) exec "$python_bin" "$root/federated/pack.py" "$@" ;;
  fed-train) exec "$python_bin" "$root/federated/run.py" "$@" ;;
  pilot)    exec "$python_bin" "$root/federated/local_lora.py" "$@" ;;
  help|-h|--help)
    printf '%s\n' 'Usage: bash run.sh <command> [options]' \
      '  agent | infer' \
      '  calibrate | he-only | he-gpu | he-multi | he-mpc | he-mpc-server' \
      '  fetch-data | build-data | revise-data | pack-train | fed-train | pilot'
    ;;
  *) printf 'Unknown command: %s\n' "$command_name" >&2; exit 2 ;;
esac
