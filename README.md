# TSRouter-VLDB

TSRouter-VLDB provides the public reproduction interface for TSRouter and TSFM-ZooBench. The package stores release metadata, artifact layout definitions, and command entry points for downloading artifacts, checking integrity, and reproducing the released main paper tables.

Large artifacts are distributed through a Hugging Face Dataset. GitHub contains source code, small configuration files, scripts, and integrity checks.

The first artifact release focuses on the stage-20 main experiment, TSRouter-main, TSRouter-fast, and paper baseline results. Ablation caches and ablation result files are outside this artifact bundle.

## Artifact Setup

Set the Hugging Face Dataset repo and local artifact root:

```bash
export TSROUTER_VLDB_HF_REPO="org/tsrouter-vldb-artifacts"
export TSROUTER_VLDB_ARTIFACT_ROOT="/path/to/TSRouter-VLDB"
```

Inspect the required bundles:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts plan --group core
```

Download the core artifact bundles:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts pull --repo-id org/tsrouter-vldb-artifacts --group core
```

Check the local release layout:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py check layout
```

Check extracted artifacts and prepare backend-readable paths:

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group core
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts prepare-backend --group core
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts prepare-backend --group core --apply
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
The backend execution contract is defined in `configs/legacy_run_contract.yaml`.
