#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RELEASE_ROOT/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="$WORKSPACE_ROOT"
ARTIFACT_ROOT="$RELEASE_ROOT"
GROUP=""
GROUP_SET="false"
WORKFLOW_MODE="fast"
REUSE_MODE="results"
LOG_DIR="$RELEASE_ROOT/reproduction_logs"
PULL_ARTIFACTS="false"
REPO_ID="${TSROUTER_VLDB_HF_REPO:-}"
CHECKPOINT_ROOT="${TSROUTER_CHECKPOINT_ROOT:-}"
DEVICES=""
QUICK_TEST="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root|--workspace-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      GROUP_SET="true"
      shift 2
      ;;
    --mode)
      WORKFLOW_MODE="$2"
      shift 2
      ;;
    --reuse)
      REUSE_MODE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --pull)
      PULL_ARTIFACTS="true"
      shift
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --devices)
      DEVICES="$2"
      shift 2
      ;;
    --quick-test)
      QUICK_TEST="true"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$REUSE_MODE" in
  results|route|core)
    ;;
  *)
    echo "unknown reuse level: $REUSE_MODE (choose results, route, or core)" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
export TSROUTER_VLDB_ARTIFACT_ROOT="$ARTIFACT_ROOT"
if [[ -z "$CHECKPOINT_ROOT" ]]; then
  CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
fi
export TSROUTER_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
CLI=("$PYTHON_BIN" "$RELEASE_ROOT/src/cli/tsrouter_vldb.py")

if [[ "$GROUP_SET" == "false" ]]; then
  GROUP="$REUSE_MODE"
fi

if [[ "$REUSE_MODE" == "route" || "$REUSE_MODE" == "core" ]]; then
  TABLES_DIR="$RUN_ROOT/results_csv/TSRouter/vldb/tables"
else
  TABLES_DIR="$ARTIFACT_ROOT/artifacts/tables_figures/results_csv/TSRouter/vldb/tables"
fi

run_step() {
  local label="$1"
  local log_file="$2"
  shift 2
  echo
  echo "== $label =="
  echo "$*"
  if "$@" > "$log_file" 2>&1; then
    echo "log: $log_file"
  else
    local status=$?
    echo "failed with exit code $status"
    echo "log: $log_file"
    if [[ -f "$log_file" ]]; then
      sed -n '1,200p' "$log_file"
    fi
    return "$status"
  fi
}

run_workflow_step() {
  local label="$1"
  local json_log="$2"
  local progress_log="$3"
  shift 3
  echo
  echo "== $label =="
  echo "$*"
  if TSROUTER_RUNTIME_LOG_DIR="$LOG_DIR/operations" TSROUTER_PROGRESS_STREAM=1 "$@" \
      > "$json_log" 2> >(tee "$progress_log" >&2); then
    echo "workflow log: $json_log"
    echo "operation logs: $LOG_DIR/operations"
  else
    local status=$?
    echo "failed with exit code $status"
    echo "workflow log: $json_log"
    echo "operation logs: $LOG_DIR/operations"
    return "$status"
  fi
}

cd "$RUN_ROOT"

echo "TSRouter-VLDB public reproduction"
echo "release root: $RELEASE_ROOT"
echo "workspace root: $RUN_ROOT"
echo "artifact root: $ARTIFACT_ROOT"
echo "artifact group: $GROUP"
echo "workflow mode: $WORKFLOW_MODE"
echo "reuse mode: $REUSE_MODE"
echo "checkpoint root: $CHECKPOINT_ROOT"
if [[ -n "$DEVICES" ]]; then
  echo "devices: $DEVICES"
fi
if [[ "$QUICK_TEST" == "true" ]]; then
  echo "benchmark mode: compact"
fi

run_step "1/8 Check release contract" "$LOG_DIR/01_release_contract.json" \
  "$PYTHON_BIN" "$RELEASE_ROOT/scripts/check_release_contract.py"

if [[ "$REUSE_MODE" == "core" ]]; then
  run_step "2/8 Check checkpoints" "$LOG_DIR/02_checkpoint_check.json" \
    "$PYTHON_BIN" "$RELEASE_ROOT/scripts/check_checkpoints.py" \
    --workspace-root "$RUN_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"
  run_step "3/8 Check GIFT-Eval" "$LOG_DIR/03_gifteval_check.json" \
    "$PYTHON_BIN" "$RELEASE_ROOT/scripts/check_gifteval.py" \
    --root "${TSROUTER_GIFTEVAL_ROOT:-$RUN_ROOT/data/gifteval}"
else
  echo
  echo "== 2/8 Check external inputs =="
  echo "skipped; selected reuse level uses released artifacts only"
fi

if [[ "$PULL_ARTIFACTS" == "true" ]]; then
  if [[ -z "$REPO_ID" ]]; then
    echo "missing artifact repository: pass --repo-id or set TSROUTER_VLDB_HF_REPO" >&2
    exit 2
  fi
  run_step "4/8 Download artifacts" "$LOG_DIR/04_artifacts_pull.json" \
    "${CLI[@]}" artifacts pull --group "$GROUP" --repo-id "$REPO_ID"
else
  echo
  echo "== 4/8 Download artifacts =="
  echo "skipped; using local artifact bundles under $RELEASE_ROOT"
fi

run_step "5/8 Check artifact archives" "$LOG_DIR/05_artifact_archives.json" \
  "${CLI[@]}" artifacts check --group "$GROUP" --skip-contents

run_step "6/8 Extract artifacts" "$LOG_DIR/06_artifact_extract.json" \
  "${CLI[@]}" artifacts extract --group "$GROUP"

run_step "7/8 Check extracted artifacts" "$LOG_DIR/07_artifact_contents.json" \
  "${CLI[@]}" artifacts check --group "$GROUP" --skip-archives

WORKFLOW_LOG="$LOG_DIR/08_workflow_${WORKFLOW_MODE}.json"
PROGRESS_LOG="$LOG_DIR/08_workflow_${WORKFLOW_MODE}.progress.log"
WORKFLOW_ARGS=(workflow run --mode "$WORKFLOW_MODE" --reuse "$REUSE_MODE" --workspace-root "$RUN_ROOT" --python-bin "$PYTHON_BIN" --execute)
if [[ -n "$DEVICES" ]]; then
  WORKFLOW_ARGS+=(--devices "$DEVICES")
fi
if [[ "$QUICK_TEST" == "true" ]]; then
  WORKFLOW_ARGS+=(--quick-test)
fi
run_workflow_step "8/8 Run workflow" "$WORKFLOW_LOG" "$PROGRESS_LOG" \
  "${CLI[@]}" "${WORKFLOW_ARGS[@]}"

"$PYTHON_BIN" "$RELEASE_ROOT/scripts/summarize_public_reproduction.py" \
  --workflow-json "$WORKFLOW_LOG" \
  --tables-dir "$TABLES_DIR"
