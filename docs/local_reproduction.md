# 本地复现手册

本文档说明 TSRouter-VLDB 发布版 artifact 的本地检查、解包、路径准备和复现流程。公开运行入口以 `TSRouter-VLDB/README.md` 和 `configs/*.yaml` 为准。

## 1. 环境

在服务器仓库根目录运行：

```bash
conda activate TSFM-py311
cd /path/to/TSRouter-v0
```

发布入口使用脚本路径，不使用 `python -m tsrouter_vldb`：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py --version
```

## 2. 本地 artifact 目录

默认 artifact root 是 `TSRouter-VLDB/`。也可以显式指定：

```bash
export TSROUTER_VLDB_ARTIFACT_ROOT="$PWD/TSRouter-VLDB"
export TSROUTER_VLDB_HF_REPO="LAMDA-shihn/tsrouter-v1-artifacts"
```

新目录中的关键路径：

| 新路径 | 用途 | 对应旧路径 |
| --- | --- | --- |
| `data/profile_sources/chronos/` | Chronos profile source | `Dataset/Repr_data_sourse/c62.tsf` |
| `data/profile_sources/moirai_timesfm/` | Moirai/TimesFM domain-stratified profile sources | `Dataset/Repr_data_sourse/{energy,nature,healthcare,web_cloudops,sales}_num1w_len992_sd2029_std.npy` |
| `artifacts/tsfm_results/results_csv/TSFM/cl_512/` | TSFM cl_512 结果 | `results_csv/TSFM/cl_512/` |
| `artifacts/tsrouter_core/results_artifacts/TSRouter/` | PROFILE/ROUTE/INSERT 中间结果 | `results_artifacts/TSRouter/` 的论文参数子集 |
| `artifacts/tsrouter_core/results_artifacts/caches/Sampled_repr_pool/` | Step1 candidate pool cache | `results_artifacts/caches/Sampled_repr_pool/c-e-n-h-w-s_x3000_in512_pl480_std_sd2029_awsfirst_pool*.pkl` |
| `artifacts/tsrouter_core/results_csv/TSRouter/` | TSRouter Step1-Step4 CSV | `results_csv/TSRouter/` 的论文参数子集 |
| `artifacts/task_sample_cache/results_artifacts/caches/GE_test_sample/` | Step4 sample cache | `results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.*` |
| `artifacts/baseline_results/` | VLDB baseline summary、insert timing 和可选离线 CSV | `results_csv/baselines/selectors/`、`results_csv/TSRouter/vldb/logs/` 与可选 `results_csv/baselines/vldb/` |
| `artifacts/tables_figures/` | 表格、图、benchmark metadata | `results_csv/TSRouter/vldb/`、`figs/vldb_results/stage20/`、`Dataset/*.csv/json` |

## 3. 本地迁移检查

先查看需要的 bundle：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts plan --group all
```

如果服务器旧路径中已经有白名单文件，可以直接 staging：

```bash
python TSRouter-VLDB/scripts/stage_local_artifacts.py --group all --mode symlink --clean
```

不支持 symlink 的文件系统使用：

```bash
python TSRouter-VLDB/scripts/stage_local_artifacts.py --group all --mode copy --clean
```

若命令返回 `missing_required`，只迁移这些缺失项。不要把宽目录整体复制进新 artifact 目录。

## 4. artifact 检查和后端挂载

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group all --skip-archives
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts prepare-backend --group all
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts prepare-backend --group all --apply
```

`prepare-backend --apply` 会把新 artifact 路径链接到旧后端脚本仍然读取的路径。已有旧路径时会标记为 `target already exists`，不会覆盖。

如果服务器生产根目录里保留了很多非论文参数结果，建议用隔离代码根目录测试，避免旧宽目录影响复现判断。生产根目录只负责 staging 白名单 artifact，隔离根目录只放代码和由 artifact 挂载出来的结果路径：

```bash
export PROD_ROOT=/path/to/TSRouter-v0
export TEST_ROOT=/path/to/TSRouter-v0-release-test

rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"

rsync -a "$PROD_ROOT/TSRouter-VLDB/" "$TEST_ROOT/TSRouter-VLDB/"
rsync -a "$PROD_ROOT/src/" "$TEST_ROOT/src/"

cd "$TEST_ROOT"
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts prepare-backend \
  --group all \
  --legacy-root "$TEST_ROOT" \
  --mode symlink \
  --apply
python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group all --skip-archives
```

隔离根目录不复制旧的 `results_csv/`、`results_artifacts/`、完整 `Dataset/` 和 `Model/`。`reuse all` 会先检查发布版 artifact，检查通过后直接使用 artifact-backed skip，不读取完整 Arrow 数据集和模型权重。`reuse none` 用于重算，需要完整 benchmark 数据和模型权重。

后续 workflow 测试也使用同一个 `--legacy-root "$TEST_ROOT"`。

## 5. 契约检查

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py check layout
python TSRouter-VLDB/scripts/check_release_contract.py
```

应核对：

- `repr_size=3000`
- `repr_v=4`
- `zoo_repr_set=c-e-n-h-w-s`
- TSRouter-main 与 TSRouter-fast 只在 `route_efficiency_mode` 及派生 route id/profile id 上不同

## 6. 快速复现

只打印计划：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode fast --reuse all --no-artifact-check
```

artifact 迁移完成后执行：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run \
  --mode fast \
  --reuse all \
  --legacy-root "$TEST_ROOT" \
  --python-bin "$(which python)" \
  --execute
```

等价脚本：

```bash
bash TSRouter-VLDB/scripts/run_local_fast.sh --legacy-root "$TEST_ROOT" --execute
```

快速复现包含 TSFM、PROFILE、TSRouter-main/fast ROUTE 和 summary table 的 artifact 复用检查。

## 7. 全流程复现

只打印计划：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode full --reuse all --no-artifact-check
```

artifact 迁移完成后执行：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run \
  --mode full \
  --reuse all \
  --legacy-root "$TEST_ROOT" \
  --python-bin "$(which python)" \
  --execute
```

等价脚本：

```bash
bash TSRouter-VLDB/scripts/run_local_full.sh --legacy-root "$TEST_ROOT" --execute
```

全流程在快速复现基础上增加 INSERT 和论文 baseline 结果。

## 8. 重新运行而非 skip

重新运行单步时先看计划，再加 `--execute`：

```bash
python TSRouter-VLDB/src/cli/tsrouter_vldb.py profile run --stage 20 --variant main,fast --reuse none
python TSRouter-VLDB/src/cli/tsrouter_vldb.py route run --stage 20 --variant main,fast --reuse none
python TSRouter-VLDB/src/cli/tsrouter_vldb.py baselines run --stage 20 --methods all --reuse none
```

重算 PROFILE、ROUTE、baseline 会消耗较长时间；公开复现优先使用 artifact-backed skip。
