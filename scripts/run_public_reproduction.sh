#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RELEASE_ROOT/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LEGACY_ROOT="$WORKSPACE_ROOT"
GROUP="all"
WORKFLOW_MODE="full"
LOG_DIR="$RELEASE_ROOT/reproduction_logs"
PULL_ARTIFACTS="false"
REPO_ID="${TSROUTER_VLDB_HF_REPO:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root|--legacy-root)
      LEGACY_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
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
CLI=("$PYTHON_BIN" "$RELEASE_ROOT/src/cli/tsrouter_vldb.py")

run_step() {
  local label="$1"
  local log_file="$2"
  shift 2
  echo
  echo "== $label =="
  echo "$*"
  "$@" > "$log_file"
  echo "log: $log_file"
}

cd "$LEGACY_ROOT"

echo "TSRouter-VLDB public reproduction"
echo "release root: $RELEASE_ROOT"
echo "workspace root: $LEGACY_ROOT"
echo "artifact group: $GROUP"
echo "workflow mode: $WORKFLOW_MODE"

run_step "1/7 Check release contract" "$LOG_DIR/01_release_contract.json" \
  "$PYTHON_BIN" "$RELEASE_ROOT/scripts/check_release_contract.py"

if [[ "$PULL_ARTIFACTS" == "true" ]]; then
  if [[ -z "$REPO_ID" ]]; then
    echo "missing artifact repository: pass --repo-id or set TSROUTER_VLDB_HF_REPO" >&2
    exit 2
  fi
  run_step "2/7 Download artifacts" "$LOG_DIR/02_artifacts_pull.json" \
    "${CLI[@]}" artifacts pull --group "$GROUP" --repo-id "$REPO_ID"
else
  echo
  echo "== 2/7 Download artifacts =="
  echo "skipped; using local artifact bundles under $RELEASE_ROOT"
fi

run_step "3/7 Check artifact archives" "$LOG_DIR/03_artifact_archives.json" \
  "${CLI[@]}" artifacts check --group "$GROUP" --skip-contents

run_step "4/7 Extract artifacts" "$LOG_DIR/04_artifact_extract.json" \
  "${CLI[@]}" artifacts extract --group "$GROUP"

run_step "5/7 Check extracted artifacts" "$LOG_DIR/05_artifact_contents.json" \
  "${CLI[@]}" artifacts check --group "$GROUP" --skip-archives

run_step "6/7 Prepare backend paths" "$LOG_DIR/06_prepare_backend.json" \
  "${CLI[@]}" artifacts prepare-backend --group "$GROUP" --legacy-root "$LEGACY_ROOT" --mode symlink --apply

WORKFLOW_LOG="$LOG_DIR/07_workflow_${WORKFLOW_MODE}.json"
run_step "7/7 Run workflow" "$WORKFLOW_LOG" \
  "${CLI[@]}" workflow run --mode "$WORKFLOW_MODE" --reuse all --legacy-root "$LEGACY_ROOT" --python-bin "$PYTHON_BIN" --execute

"$PYTHON_BIN" "$RELEASE_ROOT/scripts/summarize_public_reproduction.py" \
  --workflow-json "$WORKFLOW_LOG" \
  --tables-dir "$LEGACY_ROOT/results_csv/TSRouter/vldb/tables"
