# =============================================
# encoder_config.py
# =============================================
                         
                                           
                         
#     --repr_input_dim --repr_output_dim --repr_sub_pred_len
                                                 
                      
# =============================================

ENCODER_CONFIG = {
    "SimMTM": {
        "class_name": "SimMTMEncoder",
        "module_path": "encoder.SimMTM.SimMTM_encoder",
        "encoder_model_path": "checkpoints/encoders/SimMTM_36to128.pth",
        "scaler_path": "checkpoints/encoders/scaler_SimMTM_36to128.pkl",
        "fixed_input_dim": 36,
        "default_sub_pred_len": 48,
        "default_embedding_dim": 128
    },
    "AdvStatis": {
        "class_name": "AdvancedStatisticalEncoder",
        "module_path": "encoder.Statis.AdvStat_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 96,
        "default_embedding_dim": 128,
        # "search_ensemble_enable": True,
    },
    "TimesFM": {
        "class_name": "TimesFMEncoder",
        "module_path": "encoder.TimesFM.TimesFM_encoder",
        "encoder_model_path": "checkpoints/timesfm-models/timesfm-2.5-200m-pytorch",
        "scaler_path": None,
        "default_input_dim": 96,
        "default_sub_pred_len": 192,
        "default_embedding_dim": 512,
        "search_ensemble_enable": True,
    },
    "Chronos": {
        "class_name": "ChronosEncoder",
        "module_path": "encoder.Chronos.Chronos_encoder",
                                                                                 
        "encoder_model_path": "checkpoints/chronos-models/chronos-bolt-tiny",
                                                     
        "default_embedding_dim": 512,
        "default_input_dim": 512,
        "default_sub_pred_len": 192,
        "search_ensemble_enable": True,
    },
    "Moirai": {
        "class_name": "MoiraiEncoder",
        "module_path": "encoder.Moirai.Moirai_encoder",
        "encoder_model_path": "checkpoints/moirai-models/moirai-1.0-R-small",
        "patch_size": 128,          
        "default_input_dim": 36,
        "default_sub_pred_len": 96,
        "default_embedding_dim": 128,
        "search_ensemble_enable": True,
    },
    "RandomMLP": {
        "class_name": "RandomMLPEncoder",
        "module_path": "encoder.baseline.RandomMLP_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 36,
        "default_sub_pred_len": 96,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

                              
        "mlp_hidden_dims": [256, 256],
        "mlp_dropout": 0.1,
        "mlp_activation": "gelu",
        "mlp_pool": "mean",  # mean / last / flatten
        "mlp_l2norm": True,
        "random_stats_fusion": "none",
        "random_stats_normalize": True,
    },
    "RandomPatch": {
        "class_name": "RandomPatchEncoder",
        "module_path": "encoder.baseline.RandomTS_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

        "patch_len": 16,
        "patch_stride": 8,
        "patch_hidden_dim": 256,
        "patch_depth": 2,
        "patch_dropout": 0.0,
        "patch_activation": "gelu",
        "ts_l2norm": True,
        "ts_instance_norm": False,
        "random_stats_fusion": "none",
        "random_stats_normalize": True,
    },
    "RandomConv": {
        "class_name": "RandomConvEncoder",
        "module_path": "encoder.baseline.RandomTS_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

        "conv_channels": 128,
        "conv_kernel_size": 5,
        "conv_dilations": [1, 2, 4, 8],
        "conv_dropout": 0.0,
        "conv_activation": "gelu",
        "ts_l2norm": True,
        "ts_instance_norm": False,
        "random_stats_fusion": "none",
        "random_stats_normalize": True,
    },
    "RandomInception": {
        "class_name": "RandomInceptionEncoder",
        "module_path": "encoder.baseline.RandomTS_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

        "inception_branch_channels": 48,
        "inception_kernels": [3, 5, 9, 17],
        "inception_dropout": 0.0,
        "inception_activation": "gelu",
        "ts_l2norm": True,
        "ts_instance_norm": False,
        "random_stats_fusion": "none",
        "random_stats_normalize": True,
    },
    "RandomTCN": {
        "class_name": "RandomTCNEncoder",
        "module_path": "encoder.baseline.RandomTS_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

        "tcn_channels": 128,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4, 8, 16],
        "tcn_dropout": 0.0,
        "tcn_activation": "gelu",
        "ts_l2norm": True,
        "ts_instance_norm": False,
        "random_stats_fusion": "none",
        "random_stats_normalize": True,
    },
    "RandomFourier": {
        "class_name": "RandomFourierEncoder",
        "module_path": "encoder.baseline.RandomTS_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

        "fourier_bins": 64,
        "fourier_random_features": 128,
        "fourier_hidden_dim": 256,
        "fourier_dropout": 0.0,
        "fourier_activation": "gelu",
        "wavelet_scales": [2, 4, 8, 16, 32],
        "ts_l2norm": True,
        "ts_instance_norm": False,
        "random_stats_fusion": "none",
        "random_stats_normalize": True,
    },
    "TS2Vec": {
        "class_name": "TS2VecEncoder",
        "module_path": "encoder.baseline.TS2Vec_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,

        "ts2vec_hidden_dim": 256,
        "ts2vec_depth": 4,
        "ts2vec_kernel_size": 3,
        "ts2vec_dropout": 0.1,
        "ts_l2norm": True,
    },
    "TrainTS2Vec": {
        "class_name": "TS2VecEncoder",
        "module_path": "encoder.baseline.TS2Vec_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,
        "encoder_type": "Train",

        "ts2vec_hidden_dim": 256,
        "ts2vec_depth": 4,
        "ts2vec_kernel_size": 3,
        "ts2vec_dropout": 0.1,
        "ts_l2norm": True,
    },
    "SimpleTS2Vec": {
        "class_name": "TS2VecEncoder",
        "module_path": "encoder.baseline.TS2Vec_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 256,
        "search_ensemble_enable": True,
        "encoder_type": "SimpleTS",

        # The exact architecture is overwritten from the validated SimpleTS
        # checkpoint config before construction.
        "ts2vec_hidden_dim": 256,
        "ts2vec_depth": 4,
        "ts2vec_kernel_size": 3,
        "ts2vec_dropout": 0.1,
        "ts_l2norm": True,
    },
    "None": {
        "class_name": "NoneEncoder",
        "module_path": "encoder.baseline.None_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 512,
        "search_ensemble_enable": True,
    },
    "StatsNone": {
        "class_name": "NoneEncoder",
        "module_path": "encoder.baseline.None_encoder",
        "encoder_model_path": None,
        "scaler_path": None,
        "default_input_dim": 512,
        "default_sub_pred_len": 480,
        "default_embedding_dim": 512,
        "search_ensemble_enable": True,
        "random_stats_fusion": "early",
        "random_stats_normalize": True,
    },

    # "UniTS": {
    #     "class_name": "UnitsEncoder",
    #     "module_path": "encoder.UniTS.UniTS_encoder",
    #     "encoder_model_path": "checkpoints/units-models/units_x128_pretrain_checkpoint.pth",
    #     "scaler_path": None,
    #     "input_dim": 96,
    #     "sub_pred_len": 192,
    #     "embedding_dim": 128
    # },
}


def _register_random_stats_variants():
    bases = ["RandomMLP", "RandomPatch", "RandomConv", "RandomInception", "RandomTCN", "RandomFourier"]
    for base in bases:
        cfg = ENCODER_CONFIG[base]

        early = dict(cfg)
        early["random_stats_fusion"] = "early"
        ENCODER_CONFIG[f"Stats{base}"] = early

        late = dict(cfg)
        late["random_stats_fusion"] = "late"
        ENCODER_CONFIG[base.replace("Random", "RandomStats", 1)] = late

        train = dict(cfg)
        train["encoder_type"] = "Train"
        train["random_stats_fusion"] = "none"
        ENCODER_CONFIG[base.replace("Random", "Train", 1)] = train


_register_random_stats_variants()
