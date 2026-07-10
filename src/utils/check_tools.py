import pandas as pd
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr, kendalltau

from config.dataset_config import ALL_Fast_DATASETS, ALL_DATASETS
from config.model_zoo_config import Model_abbrev_map, Model_zoo_details


def check_results_file(csv_file_path,verbose=False,quick_test=False):
    'TSRouter runtime message.'
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"TSRouter runtime message: {e}")
        return None

    if verbose:
        print(f"\n{'=' * 50}TSRouter runtime message: {csv_file_path}",end=" ")

               
    check_dataset_completeness(df, verbose,quick_test)
           
    df = check_duplicate_results(df, csv_file_path, verbose)
               
    check_model_naming(df, verbose)
                   
    analyze_model_results(df, verbose)

    return df

def check_dataset_completeness(df,verbose,quick_test):
    'TSRouter runtime message.'
    done_datasets = set(df["dataset"].unique())
    if quick_test:
        all_datasets = set(ALL_Fast_DATASETS)
    else:
        all_datasets = set(ALL_DATASETS)

            
    missing_datasets = all_datasets - done_datasets
    if missing_datasets:
        if verbose:
            print(f"TSRouter runtime message: {len(missing_datasets)}TSRouter runtime message: ")
            for dataset in sorted(missing_datasets):
                print(f"  - {dataset}", end=" ")
            print()
    else:
        if verbose:
            print(f"✅ {len(all_datasets)}TSRouter runtime message: ",end=" ")

             
    extra_datasets = done_datasets - all_datasets
    if extra_datasets:
        if verbose:
            print(f"TSRouter runtime message: {len(extra_datasets)}TSRouter runtime message: ")
            # for dataset in sorted(extra_datasets):
            #     print(f"  - {dataset}",end=" ")
            # print()



def analyze_model_results(df,verbose,verbose_grouped=False):
    'TSRouter runtime message.'
    if not verbose:
        return

    df = df.copy()
    if "dataset" not in df.columns:
        print('TSRouter runtime message.')
        return
             
    df[['ds_key', 'ds_freq', 'term']] = df['dataset'].str.extract(r'^(.*?)/([^/]+)/([^/]+)$')

                 
    if "MASE" not in df.columns and "eval_metrics/MASE[0.5]" in df.columns:
        df["MASE"] = pd.to_numeric(df["eval_metrics/MASE[0.5]"], errors="coerce")
    if "sMAPE" not in df.columns and "eval_metrics/sMAPE[0.5]" in df.columns:
        df["sMAPE"] = pd.to_numeric(df["eval_metrics/sMAPE[0.5]"], errors="coerce")
    if "CRPS" not in df.columns and "eval_metrics/mean_weighted_sum_quantile_loss" in df.columns:
        df["CRPS"] = pd.to_numeric(df["eval_metrics/mean_weighted_sum_quantile_loss"], errors="coerce")

            
    metrics = ['sMAPE', 'MASE', 'CRPS']
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print('TSRouter runtime message.')
        return

    print("df shape:", df.shape, end="  ")
    print('TSRouter runtime message.', end=" ")
    for metric in metrics:
        avg = df[metric].mean()
        print(f"{metric}: {avg:.4f}", end=" ")
    print()

    if verbose_grouped:
               
        all_freqs = sorted(df['ds_freq'].dropna().unique())
        all_terms = sorted(df['term'].dropna().unique())
        all_domains = sorted(df['domain'].dropna().unique())

                    
        def group_and_print(group_col, group_name, group_keys):
            metric_dict = {metric: [df[df[group_col] == key][metric].mean() for key in group_keys] for metric in metrics}
            _print_simple_table(metric_dict, group_name, group_keys)

                   
        def _print_simple_table(metric_dict, group_name, groups):
            print(f"TSRouter runtime message: {group_name}TSRouter runtime message: ")
            header = f"{'':<8}" + "".join([f"{g:<10}" for g in groups])
            print(header)
            for metric, values in metric_dict.items():
                row = f"{metric:<8}"
                for v in values:
                    if pd.isna(v):
                        row += f"{'':<10}"
                    else:
                        row += f"{v:.4f}".ljust(10)
                print(row)
        group_and_print('ds_freq', 'TSRouter runtime message.', all_freqs)
        group_and_print('term', 'TSRouter runtime message.', all_terms)
        group_and_print('domain', 'TSRouter runtime message.', all_domains)
        group_and_print('num_variates', 'TSRouter runtime message.', sorted(df['num_variates'].dropna().unique()))

def check_duplicate_results(df: pd.DataFrame, csv_file_path, verbose: bool = False) -> pd.DataFrame:
    'TSRouter runtime message.'
    df_cleaned = df.copy()

    dataset_counts = df_cleaned["dataset"].value_counts()
    duplicate_datasets = dataset_counts[dataset_counts > 1].index.tolist()

    if not duplicate_datasets:
        return df_cleaned

    print(f"TSRouter runtime message: {len(duplicate_datasets)}TSRouter runtime message: ", end=" ")

    removed_datasets = []
    needs_save = False

             
    metric_candidates = ["MASE", "eval_metrics/MASE[0.5]"]
    metric_col = next((c for c in metric_candidates if c in df_cleaned.columns), None)
    if metric_col is None:
        if verbose:
            print('TSRouter runtime message.')
        return df_cleaned

    for dataset in duplicate_datasets:
        dup_rows = df_cleaned[df_cleaned["dataset"] == dataset]

        is_consistent = True
        col_values = dup_rows[metric_col].dropna().astype(float)
        if not col_values.empty and col_values.max() - col_values.min() > 1e-3:
            is_consistent = False

        if not is_consistent:
            print(f" ⚠️  - {dataset}: {len(dup_rows)} duplicated rows differ; kept the latest row.")
        # Result files are append/update oriented; the latest row is authoritative.
        keep_index = dup_rows.index[-1]
        to_remove = dup_rows.index[dup_rows.index != keep_index]
        df_cleaned = df_cleaned.drop(to_remove)

        removed_datasets.append((dataset, len(to_remove), "keep_last", is_consistent))
        needs_save = True

    if removed_datasets:
        print('TSRouter runtime message.', removed_datasets)
    if needs_save:
        try:
            df_cleaned.to_csv(csv_file_path, index=False)
            print(f"TSRouter runtime message: {csv_file_path}")
        except Exception as e:
            print(f"TSRouter runtime message: {e}")

    return df_cleaned


def check_model_naming(df,verbose):
    'TSRouter runtime message.'

    model_names = df["model"].unique()

    if len(model_names) <= 1:
        return

    print(f"TSRouter runtime message: {model_names}TSRouter runtime message: ")

                       
    model_mix = df.groupby("dataset")["model"].unique()
    mixed_datasets = model_mix[model_mix.apply(len) > 1]

    if not mixed_datasets.empty:
        print('TSRouter runtime message.')
        for dataset, names in mixed_datasets.items():
            print(f"  - {dataset}: {names}")



def standardize_model_names(baseline_data, model_col: str = "model") -> pd.DataFrame:
    'TSRouter runtime message.'
                    
    baseline_df = pd.concat(baseline_data, ignore_index=True)
    baseline_df[['ds_key', 'ds_freq', 'term']] = baseline_df['dataset'].str.extract(r'^(.*?)/([^/]+)/([^/]+)$')
    def _normalize(name: str) -> str:
        parts = name.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}_{parts[1]}"
        return name

    df = baseline_df.copy()
    df[model_col] = df[model_col].apply(
        lambda x: Model_abbrev_map.get(_normalize(str(x)), x)
    )
    return df

def calculate_order_metrics(df_real, df_pred, k=None):
    'TSRouter runtime message.'

    if k is None:
        k_list: list[int] = []
    elif isinstance(k, int):
        k_list = [k]
    else:
        k_list = list(k)

    spearman_vals = []
    kendall_vals = []
    acc_top = {kk: [] for kk in k_list}
    real1_in_pred = {kk: [] for kk in k_list}
    pred1_in_real = {kk: [] for kk in k_list}

    common_datasets = set(df_real["dataset"]).intersection(set(df_pred["dataset"]))

    for dataset in common_datasets:
        real_order = df_real[df_real["dataset"] == dataset]["model_order"].iloc[0]
        pred_order = df_pred[df_pred["dataset"] == dataset]["model_order"].iloc[0]

        if real_order is None or pred_order is None:
            continue
        if len(real_order) == 0 or len(pred_order) == 0:
            continue
        if len(real_order) != len(pred_order):
                                   
            min_len = min(len(real_order), len(pred_order))
            real_order = real_order[:min_len]
            pred_order = pred_order[:min_len]

                                            
                                              
        items = list(dict.fromkeys(real_order))        

        real_pos = {m: i for i, m in enumerate(real_order)}
        pred_pos = {m: i for i, m in enumerate(pred_order)}

        real_rank = [real_pos[m] for m in items]
        pred_rank = [pred_pos[m] for m in items]

        corr, _ = spearmanr(real_rank, pred_rank)
        tau, _ = kendalltau(real_rank, pred_rank)

        spearman_vals.append(corr)
        kendall_vals.append(tau)
        spearman_vals.append(corr)
        kendall_vals.append(tau)

                                 
        for kk in k_list:
                         
            kk_eff = min(kk, len(real_order), len(pred_order))
            if kk_eff <= 0:
                continue

            real_k = real_order[:kk_eff]
            pred_k = pred_order[:kk_eff]

                           
            correct = len(set(real_k) & set(pred_k)) / kk_eff
            acc_top[kk].append(correct)

                                                  
            real1_in_pred[kk].append(1 if real_order[0] in pred_k else 0)

                                                  
            pred1_in_real[kk].append(1 if pred_order[0] in real_k else 0)

    result = {}

    result["Spearman"] = np.nanmean(spearman_vals) if spearman_vals else np.nan
    result["KendallTau"] = np.nanmean(kendall_vals) if kendall_vals else np.nan

    for kk in k_list:
        if acc_top[kk]:
            result[f"Acc_TopK{kk}"] = np.nanmean(acc_top[kk])
            result[f"Real1_in_PredK{kk}"] = np.nanmean(real1_in_pred[kk])
            result[f"Pred1_in_RealK{kk}"] = np.nanmean(pred1_in_real[kk])
        else:
            result[f"Acc_TopK{kk}"] = np.nan
            result[f"Real1_in_PredK{kk}"] = np.nan
            result[f"Pred1_in_RealK{kk}"] = np.nan

    return result

def filter_models_by_key(model_zoo, select_date, select_key: str = "release_date"):
    'TSRouter runtime message.'
    all_models = []
    for family, sizes in model_zoo.items():
        for size, details in sizes.items():
            d1 = datetime.strptime(details[select_key], "%Y-%m-%d")
            d2 = datetime.strptime(select_date, "%Y-%m-%d")
            if d1 <= d2:
                details_with_meta = details.copy()
                details_with_meta["_family"] = family
                details_with_meta["_size"] = size
                all_models.append(details_with_meta)

    all_models_sorted = sorted(all_models, key=lambda x: x["release_date"])

    filtered_zoo = {}
    for idx, model in enumerate(all_models_sorted):
        family = model["_family"]
        size = model["_size"]
        model_details = {k: v for k, v in model.items() if not k.startswith("_")}
        model_details["id"] = idx

        if family not in filtered_zoo:
            filtered_zoo[family] = {}
        filtered_zoo[family][size] = model_details

    filtered_models = [
        details for family in filtered_zoo.values() for details in family.values()
    ]
    sorted_filtered_models = sorted(filtered_models, key=lambda x: x["id"])

    return filtered_zoo, sorted_filtered_models



