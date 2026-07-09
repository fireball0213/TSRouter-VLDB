# TSRouter-VLDB

TSRouter-VLDB provides the public reproduction interface for TSRouter and TSFM-ZooBench. The package stores release metadata, artifact layout definitions, and command entry points for downloading artifacts, checking integrity, and reproducing the released main paper tables.

Large artifacts are distributed through a Hugging Face Dataset. GitHub contains source code, small configuration files, scripts, and integrity checks.

The artifact release provides the main-paper reproduction bundles for TSRouter-main, TSRouter-fast, and paper baselines.

## Artifact Setup

Set the Hugging Face Dataset repo and local artifact root:

```bash
export TSROUTER_VLDB_HF_REPO="LAMDA-shihn/tsrouter-v1-artifacts"
export TSROUTER_VLDB_ARTIFACT_ROOT="/path/to/TSRouter-VLDB"
```

If artifact bundles are already present under `TSRouter-VLDB/`, the repository variable is not required.

Inspect the required bundles:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts plan --group core
```

Download the core artifact bundles:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts pull --repo-id "$TSROUTER_VLDB_HF_REPO" --group core
```

Check the local release layout:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py check layout
```

Check and extract the local artifact bundles:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group core --skip-contents
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts extract --group core
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group core --skip-archives
```

Check that the release execution contract matches the paper main grid and command plans:

```bash
python TSRouter-VLDB/scripts/check_release_contract.py
```

## Public Commands

The public command groups use paper-facing names:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py tsfm run --stage 20 --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py profile run --stage 20 --variant main,fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py route run --stage 20 --variant main,fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py insert run-all --start-stage 3 --end-stage 20 --variant main,fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py baselines run --stage 20 --methods all --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py summary tables --stage 20 --write
```

Commands print a dry execution plan by default. Add `--execute` to run the generated backend commands after checking the plan.

## Workflows

One-command public reproduction with readable progress and table previews:

```bash
bash TSRouter-VLDB/scripts/run_public_reproduction.sh \
  --root "$PWD" \
  --python-bin "$(which python)" \
  --mode full
```

If the artifact bundles are not already present under `TSRouter-VLDB/`, add `--pull --repo-id "$TSROUTER_VLDB_HF_REPO"`.
The script writes full JSON logs to `TSRouter-VLDB/reproduction_logs/` and prints a concise workflow summary plus released table previews.

Artifact repository: https://huggingface.co/datasets/LAMDA-shihn/tsrouter-v1-artifacts

Fast artifact-backed reproduction:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode fast --reuse all --execute
```

Full artifact-backed reproduction:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode full --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode full --reuse all --execute
```

The artifact layout is defined in `configs/artifact_layout.yaml`.
