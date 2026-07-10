# selector/selector_config.py
                                      

Selector_zoo_details = {
    "Random_Select": {
                                                                                           
        "model_module": "selector.baselines.baseline_select",
                     
        "model_class": "Random_Select_Model",
                  
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_ss{search_seed}_"
            "all_results.csv"
        ),
    },

    "All_Select": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "All_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "all_ensemble_{ensemble_agg}_"
            "all_results.csv"
        ),
    },

    "Real_Select": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Real_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_real_{real_order_metric}_"
            "all_results.csv"
        ),
    },

    "Real_Channel_Select": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Real_Channel_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_real_channel_{real_order_metric}_"
            "all_results.csv"
        ),
    },

    "Recent_Select": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Recent_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_recent_"
            "all_results.csv"
        ),
    },

    "Task_Probe_Forward_Select": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Task_Probe_Forward_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_task_probe_forward_task{sample_repr_num}_"
            "all_results.csv"
        ),
    },

    "Current_best_sMAPE_Rank": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Current_best_sMAPE_Rank_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_current_best_sMAPE_Rank_"
            "all_results.csv"
        ),
    },

    "Current_best_sMAPE": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Current_best_sMAPE_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_current_best_sMAPE_"
            "all_results.csv"
        ),
    },

    "Current_best_MASE": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Current_best_MASE_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_current_best_MASE_"
            "all_results.csv"
        ),
    },

    "Current_best_CRPS": {
        "model_module": "selector.baselines.baseline_select",
        "model_class": "Current_best_CRPS_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_current_best_CRPS_"
            "all_results.csv"
        ),
    },
    "LogME_Select": {
        "model_module": "selector.baselines.logme_select",
        "model_class": "LogME_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_logme_"
            "all_results.csv"
        ),
    },


    "TSRouter": {
        "model_module": "selector.TSRouter_Select.tsrouter_select",
        "model_class": "TSRouter_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}-{ensemble_agg}_TSRouter_"
            "all_results.csv"
        ),
    },

}
