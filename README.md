# TSRouter-VLDB

This repository contains the public implementation and reproduction package for TSRouter and TSFM-ZooBench.

TSFM-ZooBench evaluates time-series foundation models in a service setting where the available model zoo grows over time. It uses the public [GIFT-Eval benchmark](https://huggingface.co/datasets/Salesforce/GiftEval) as the forecasting workload and records model quality, runtime, and routing evidence for model-selection experiments.

TSRouter profiles the capability of available TSFMs, routes forecasting requests to a suitable model, and updates routing evidence as new models or results are added. The repository includes the method implementation, paper baseline implementations, workflow commands, and reproduction materials.

## Artifacts

[**TSRouter-VLDB Artifacts on Hugging Face**](https://huggingface.co/datasets/LAMDA-shihn/tsrouter-v1-artifacts)

The artifact repository provides released result records, capability representations, request-sample caches, profile inputs, and table inputs. GIFT-Eval remains available from its [official Hugging Face repository](https://huggingface.co/datasets/Salesforce/GiftEval).

## Setup

Clone the repository and enter its root directory:

```bash
git clone https://github.com/fireball0213/TSRouter-VLDB.git
cd TSRouter-VLDB
```

Install the lightweight dependencies for artifact checks and table preview:

```bash
python -m pip install -r requirements_core.txt
```

Install the method dependencies for TSFM and selector components:

```bash
python -m pip install -r requirements_method.txt
```

The released workflows use the appropriate inputs and results for their selected reproduction level. When a workflow requires local checkpoints or GIFT-Eval, its preflight check reports the required location.

On a connected machine, prefetch the official model checkpoints and benchmark:

```bash
python scripts/fetch_model_weights.py --out "$PWD/checkpoints"
python scripts/fetch_gifteval.py --out "$PWD/data/gifteval"
```

Use `--model <family_variant>` with `fetch_model_weights.py` to retrieve selected checkpoints. The script records the downloaded upstream revisions in `checkpoints/checkpoint_manifest.json`. Access controls and license terms of each upstream model repository remain applicable.

Set the artifact repository:

```bash
export TSROUTER_VLDB_HF_REPO="LAMDA-shihn/tsrouter-v1-artifacts"
```

## Reproduction Levels

The `--reuse` level selects a reproducible scope and its required inputs:

| Level | Recomputed work | Required downloads |
| --- | --- | --- |
| `results` | Artifact validation and table preview | TSRouter artifact repository only |
| `route` | TSRouter main and fast routing from released capability representations and request-sample caches | TSRouter artifact repository only |
| `core` | TSRouter capability profiling followed by main and fast routing | TSRouter artifact repository, [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval), and the official TSFM checkpoints |

The `results` level is the fastest way to inspect the released paper outputs. The `route` level is the recommended quick method check: it reuses released capability representations, pooled inputs, request-sample caches, and TSFM metric records, then recomputes the routing decisions without loading benchmark data or model checkpoints. The `core` level rebuilds the capability representations with the public benchmark and official checkpoints, while reusing the released TSFM metric records and request-sample cache.

All levels write JSON logs to `reproduction_logs/`. Commands that produce tables write them to `results_csv/TSRouter/vldb/tables/`.

## Results Check

To validate the released result package and preview the tables:

```bash
bash scripts/run_public_reproduction.sh \
  --root "$PWD" \
  --python-bin "$(which python)" \
  --reuse results \
  --pull \
  --repo-id "$TSROUTER_VLDB_HF_REPO"
```

## Route Check

To rerun the TSRouter main and fast route decisions from released representations and request samples:

```bash
bash scripts/run_public_reproduction.sh \
  --root "$PWD" \
  --python-bin "$(which python)" \
  --reuse route \
  --pull \
  --repo-id "$TSROUTER_VLDB_HF_REPO"
```

## Core Check

Download the public benchmark and official checkpoints on a connected machine:

```bash
python scripts/fetch_gifteval.py --out "$PWD/data/gifteval"
python scripts/fetch_model_weights.py --out "$PWD/checkpoints"
```

Then rebuild TSRouter capability representations and route decisions:

```bash
bash scripts/run_public_reproduction.sh \
  --root "$PWD" \
  --checkpoint-root "$PWD/checkpoints" \
  --python-bin "$(which python)" \
  --reuse core \
  --pull \
  --repo-id "$TSROUTER_VLDB_HF_REPO"
```

## Command Groups

The workflow can also be run one command group at a time:

```bash
python src/cli/tsrouter_vldb.py profile run --stage 20 --variant main,fast --reuse core --execute --workspace-root "$PWD" --python-bin "$(which python)"
python src/cli/tsrouter_vldb.py route run --stage 20 --variant main,fast --reuse route --execute --workspace-root "$PWD" --python-bin "$(which python)"
python src/cli/tsrouter_vldb.py summary tables --stage 20 --reuse results --execute --workspace-root "$PWD" --python-bin "$(which python)"
```

Use `--reuse results` to validate released outputs, `--reuse route` for cache-driven route checks, and `--reuse core` when rebuilding the capability profile.
