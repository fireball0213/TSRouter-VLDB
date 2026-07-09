#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RELEASE_ROOT/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="$WORKSPACE_ROOT"
ARTIFACT_ROOT="$RELEASE_ROOT"
GROUP="all"
WORKFLOW_MODE="full"
REUSE_MODE="all"
LOG_DIR="$RELEASE_ROOT/reproduction_logs"
PULL_ARTIFACTS="false"
REPO_ID="${TSROUTER_VLDB_HF_REPO:-}"

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
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$LOG_DIR"
export TSROUTER_VLDB_ARTIFACT_ROOT="$ARTIFACT_ROOT"
CLI=("$PYTHON_BIN" "$RELEASE_ROOT/src/cli/tsrouter_vldb.py")

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

cd "$RUN_ROOT"

echo "TSRouter-VLDB public reproduction"
echo "release root: $RELEASE_ROOT"
echo "workspace root: $RUN_ROOT"
echo "artifact root: $ARTIFACT_ROOT"
echo "artifact group: $GROUP"
echo "workflow mode: $WORKFLOW_MODE"
echo "reuse mode: $REUSE_MODE"

run_step "1/6 Check release contract" "$LOG_DIR/01_release_contract.json" \
  "$PYTHON_BIN" "$RELEASE_ROOT/scripts/check_release_contract.py"

if [[ "$PULL_ARTIFACTS" == "true" ]]; then
  if [[ -z "$REPO_ID" ]]; then
    echo "missing artifact repository: pass --repo-id or set TSROUTER_VLDB_HF_REPO" >&2
    exit 2
  fi
  run_step "2/6 Download artifacts" "$LOG_DIR/02_artifacts_pull.json" \
    "${CLI[@]}" artifacts pull --group "$GROUP" --repo-id "$REPO_ID"
else
  echo
  echo "== 2/6 Download artifacts =="
  echo "skipped; using local artifact bundles under $RELEASE_ROOT"
fi

run_step "3/6 Check artifact archives" "$LOG_DIR/03_artifact_archives.json" \
  "${CLI[@]}" artifacts check --group "$GROUP" --skip-contents

run_step "4/6 Extract artifacts" "$LOG_DIR/04_artifact_extract.json" \
  "${CLI[@]}" artifacts extract --group "$GROUP"

run_step "5/6 Check extracted artifacts" "$LOG_DIR/05_artifact_contents.json" \
  "${CLI[@]}" artifacts check --group "$GROUP" --skip-archives

WORKFLOW_LOG="$LOG_DIR/06_workflow_${WORKFLOW_MODE}.json"
run_step "6/6 Run workflow" "$WORKFLOW_LOG" \
  "${CLI[@]}" workflow run --mode "$WORKFLOW_MODE" --reuse "$REUSE_MODE" --workspace-root "$RUN_ROOT" --python-bin "$PYTHON_BIN" --execute

"$PYTHON_BIN" "$RELEASE_ROOT/scripts/summarize_public_reproduction.py" \
  --workflow-json "$WORKFLOW_LOG" \
  --tables-dir "$ARTIFACT_ROOT/artifacts/tables_figures/results_csv/TSRouter/vldb/tables"
