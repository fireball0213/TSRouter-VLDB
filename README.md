# TSRouter-VLDB

TSRouter-VLDB contains the public artifact package and reproduction tools for TSRouter and TSFM-ZooBench.

TSFM-ZooBench is a benchmark for studying time-series foundation model service when the available model zoo grows over time. It records model quality, runtime, and routing evidence so that a serving method can be evaluated under realistic model-arrival and request workloads.

TSRouter is a training-free routing method for this setting. It profiles the capability of available TSFMs, routes each forecasting request to a suitable model, and updates the routing evidence when new models or results are added. This repository focuses on the released paper artifacts: integrity checks, artifact-backed reproduction, and table previews for the main experimental results.

Large artifacts are distributed through a Hugging Face Dataset. GitHub contains source code, small configuration files, scripts, and integrity checks.

Artifact repository: https://huggingface.co/datasets/LAMDA-shihn/tsrouter-v1-artifacts

## Setup

Install the lightweight reproduction dependencies:

```bash
python -m pip install -r TSRouter-VLDB/requirements_core.txt
```

Set the Hugging Face Dataset repository when artifacts are not already present locally:

```bash
export TSROUTER_VLDB_HF_REPO="LAMDA-shihn/tsrouter-v1-artifacts"
```

Inspect, download, and verify the artifact bundles:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts plan --group all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts pull --repo-id "$TSROUTER_VLDB_HF_REPO" --group all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group all --skip-contents
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts extract --group all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group all --skip-archives
```

## Reproduction

Run the public artifact-backed workflow:

```bash
bash TSRouter-VLDB/scripts/run_public_reproduction.sh \
  --root "$PWD" \
  --python-bin "$(which python)" \
  --mode full \
  --reuse all
```

If the bundles are not already present under `TSRouter-VLDB/`, add:

```bash
--pull --repo-id "$TSROUTER_VLDB_HF_REPO"
```

The script verifies the release layout, extracts the bundles, executes the public workflow with artifact reuse, and prints the released table previews. JSON logs are written to `TSRouter-VLDB/reproduction_logs/`.

## Command Groups

The same workflow can be inspected or run one command group at a time:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py tsfm run --stage 20 --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py profile run --stage 20 --variant main,fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py route run --stage 20 --variant main,fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py insert run-all --start-stage 3 --end-stage 20 --variant main,fast --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py baselines run --stage 20 --methods all --reuse all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py summary tables --stage 20 --write
```

Commands print a compact operation plan by default. Add `--execute` to run the selected group.

## Reuse Modes

`--reuse all` is the supported public reproduction mode for the released artifact package. It checks the downloaded artifacts and skips expensive recomputation while still validating the workflow and result tables.

`--reuse none` runs the selected command group without artifact-backed skip. It requires the full local experiment environment and raw inputs used by the paper experiments, which are not part of this lightweight artifact package.

## Release Checks

Before publishing a code update, run:

```bash
python TSRouter-VLDB/scripts/check_release_contract.py
python TSRouter-VLDB/scripts/audit_public_surface.py
python -m compileall TSRouter-VLDB/src TSRouter-VLDB/scripts
bash -n TSRouter-VLDB/scripts/run_public_reproduction.sh
```

The artifact layout is defined in `configs/artifact_layout.yaml`; the public execution contract is defined in `configs/execution_contract.yaml`.
