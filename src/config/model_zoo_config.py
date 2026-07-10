
Model_zoo_details = {

    "moirai": {
        "small": {
            "name": "1.0-R-small",
            "abbreviation": "Moi.S",
            "model_module": "model_zoo.Moirai_model",
            "model_class": "MoiraiModel",
            "module_name": "Salesforce/moirai-1.0-R-small",
            "model_local_path": "checkpoints/moirai-models/moirai-1.0-R-small",
            "release_date": "2024-03-19",
        },
        "base": {
            "name": "1.0-R-base",
            "abbreviation": "Moi.B",
            "model_module": "model_zoo.Moirai_model",
            "model_class": "MoiraiModel",
            "module_name": "Salesforce/moirai-1.0-R-base",
            "model_local_path": "checkpoints/moirai-models/moirai-1.0-R-base",
            "release_date": "2024-03-20",
        },
        "large": {
            "name": "1.0-R-large",
            "abbreviation": "Moi.L",
            "model_module": "model_zoo.Moirai_model",
            "model_class": "MoiraiModel",
            "module_name": "Salesforce/moirai-1.0-R-large",
            "model_local_path": "checkpoints/moirai-models/moirai-1.0-R-large",
            "release_date": "2024-03-21",
        },
    },

    # "visionts": {
    #     "base": {
    #         "name": "base",
    #         "abbreviation": "Vis.B",
    #         "model_module": "model_zoo.VisionTS_model",
    #         "model_class": "VisionTSModel",
    #         "module_name": "Keytoyze/VisionTS",
    #         "model_local_path": "checkpoints/visionts-models",
    #         "release_date": "2024-08-25",
    #
    #     },
    # },

    "chronos": {
        "bolt_tiny": {
            "name": "bolt-tiny",
            "abbreviation": "Chr.bT",
            "model_module": "model_zoo.Chronos_model",
            "model_class": "ChronosModel",
            "module_name": "amazon/chronos-bolt-tiny",
            "model_local_path": "checkpoints/chronos-models/chronos-bolt-tiny",
            "release_date": "2024-11-10",
        },
        "bolt_mini": {
            "name": "bolt-mini",
            "abbreviation": "Chr.bM",
            "model_module": "model_zoo.Chronos_model",
            "model_class": "ChronosModel",
            "module_name": "amazon/chronos-bolt-mini",
            "model_local_path": "checkpoints/chronos-models/chronos-bolt-mini",
            "release_date": "2024-11-11",
        },
        "bolt_small": {
            "name": "bolt-small",
            "abbreviation": "Chr.bS",
            "model_module": "model_zoo.Chronos_model",
            "model_class": "ChronosModel",
            "module_name": "amazon/chronos-bolt-small",
            "model_local_path": "checkpoints/chronos-models/chronos-bolt-small",
            "release_date": "2024-11-12",
        },
        "bolt_base": {
            "name": "bolt-base",
            "abbreviation": "Chr.bB",
            "model_module": "model_zoo.Chronos_model",
            "model_class": "ChronosModel",
            "module_name": "amazon/chronos-bolt-base",
            "model_local_path": "checkpoints/chronos-models/chronos-bolt-base",
            "release_date": "2024-11-13",
        },
    },


    # "sundial": {
    #     "base": {
    #         "name": "base",
    #         "abbreviation": "Sun.B",
    #         "model_module": "model_zoo.Sundial_model",
    #         "model_class": "SundialModel",
    #         "module_name": "thuml/Sundial-base_128m",
    #         "model_local_path": "checkpoints/sundial-models/Sundial-base_128m",
    #         "release_date": "2025-05-14",
    #     },
    # },

    "moirai2": {
        "small": {
            "name": "2.0-R-small",
            "abbreviation": "Moi2.S",
            "model_module": "model_zoo.Moirai2_model",
            "model_class": "Moirai2Model",
            "module_name": "Salesforce/moirai-2.0-R-small",
            "model_local_path": "checkpoints/moirai-models/moirai-2.0-R-small",
            "release_date": "2025-08-06",
        },
    },

    "flowstate": {
        "r1": {
            "name": "flowstate",
            "abbreviation": "Flo.r1",
            "model_module": "model_zoo.FlowState_model",
            "model_class": "FlowStateModel",
            "module_name": "ibm-granite/granite-timeseries-flowstate-r1",
            "model_local_path": "checkpoints/flowstate-models",
            "release_date": "2025-09-12",
        },
    },

    "kairos": {
        "10m": {
            "name": "Kairos-10M",
            "abbreviation": "Kai.10",
            "model_module": "model_zoo.Kairos_model",
            "model_class": "KairosModel",
            "module_name": "mldi-lab/Kairos_10m",
            "model_local_path": "checkpoints/kairos-models/Kairos_10m",
            "release_date": "2025-09-28",
        },
        "23m": {
            "name": "Kairos-23M",
            "abbreviation": "Kai.23",
            "model_module": "model_zoo.Kairos_model",
            "model_class": "KairosModel",
            "module_name": "mldi-lab/Kairos_23m",
            "model_local_path": "checkpoints/kairos-models/Kairos_23m",
            "release_date": "2025-09-29",
        },
        "50m": {
            "name": "Kairos-50M",
            "abbreviation": "Kai.50",
            "model_module": "model_zoo.Kairos_model",
            "model_class": "KairosModel",
            "module_name": "mldi-lab/Kairos_50m",
            "model_local_path": "checkpoints/kairos-models/Kairos_50m",
            "release_date": "2025-09-30",
        },
    },
    "timesfm": {
        "2.5": {
            "name": "2.5-200m-pytorch",
            "abbreviation": "TFM.25",
            "model_module": "model_zoo.TimesFM_model",
            "model_class": "TimesFMModel",
            "module_name": "google/timesfm-2.5-200m-pytorch",
            "model_local_path": "checkpoints/timesfm-models/timesfm-2.5-200m-pytorch",
            "release_date": "2025-10-01",
        },
    },



    "chronos2": {
        "base": {
            "name": "base",
            "abbreviation": "Chr.2",
            "model_module": "model_zoo.Chronos2_model",
            "model_class": "Chronos2Model",
            "module_name": "amazon/chronos-2",
            "model_local_path": "checkpoints/chronos-models/chronos-2",
            "release_date": "2025-10-30",
        },
    },



    "patchtst": {
        "r1": {
            "name": "patchtst-fm-r1",
            "abbreviation": "PTS.FM",
            "model_module": "model_zoo.PatchTST_FM_model",
            "model_class": "PatchTSTFMModel",
            "module_name": "ibm-granite/granite-timeseries-patchtst-fm-r1",
            "model_local_path": "checkpoints/patchtst-models/patchtst-fm-r1/",
            "release_date": "2026-03-18",
        },
    },
    "timemoe": {
            "50m": {
                "name": "50M",
                "abbreviation": "TMoE.50",
                "model_module": "model_zoo.TimeMOE_model",
                "model_class": "TimeMOEModel",
                "module_name": "Maple728/TimeMoE-50M",
                "model_local_path": "checkpoints/timemoe-models/TimeMoE-50M/",
                "release_date": "2024-09-21",
            },
        },
    "toto": {
        "base": {
            "name": "Toto-Open-Base-1.0",
            "abbreviation": "Toto",
            "model_module": "model_zoo.Toto_model",
            "model_class": "TotoModel",
            "module_name": "Datadog/Toto-Open-Base-1.0",
            "model_local_path": "checkpoints/toto-models/Toto-Open-Base-1.0",
            "release_date": "2025-05-06",
        },
    },
    "tirex": {
        "1.1": {
            "name": "TiRex-1.1-gifteval",
            "abbreviation": "TiRex",
            "model_module": "model_zoo.TiRex_model",
            "model_class": "TiRexModel",
            "module_name": "NX-AI/TiRex-1.1-gifteval",
            "model_local_path": "checkpoints/tirex-models/TiRex-1.1-gifteval",
            "release_date": "2025-10-06",
        },
    },

    "toto2": {
        "4m": {
            "name": "Toto-2.0-4m",
            "abbreviation": "Toto2.T",
            "model_module": "model_zoo.Toto2_model",
            "model_class": "Toto2Model",
            "module_name": "Datadog/Toto-2.0-4m",
            "model_local_path": "checkpoints/toto-models/Toto-2.0-4m",
            "release_date": "2026-04-14",
        },
        "22m": {
            "name": "Toto-2.0-22m",
            "abbreviation": "Toto2.S",
            "model_module": "model_zoo.Toto2_model",
            "model_class": "Toto2Model",
            "module_name": "Datadog/Toto-2.0-22m",
            "model_local_path": "checkpoints/toto-models/Toto-2.0-22m",
            "release_date": "2026-04-15",
        },
        # "313m": {
        #     "name": "Toto-2.0-313m",
        #     "abbreviation": "Toto2.B",
        #     "model_module": "model_zoo.Toto2_model",
        #     "model_class": "Toto2Model",
        #     "module_name": "Datadog/Toto-2.0-313m",
        #     "model_local_path": "checkpoints/toto-models/Toto-2.0-313m",
        #     "release_date": "2026-05-12",
        # },
        # "1B": {
        #     "name": "Toto-2.0-1B",
        #     "abbreviation": "Toto2.L",
        #     "model_module": "model_zoo.Toto2_model",
        #     "model_class": "Toto2Model",
        #     "module_name": "Datadog/Toto-2.0-1B",
        #     "model_local_path": "checkpoints/toto-models/Toto-2.0-1B",
        #     "release_date": "2026-05-13",
        # },
        # "2.5B": {
        #     "name": "Toto-2.0-2.5B",
        #     "abbreviation": "Toto2.H",
        #     "model_module": "model_zoo.Toto2_model",
        #     "model_class": "Toto2Model",
        #     "module_name": "Datadog/Toto-2.0-2.5B",
        #     "model_local_path": "checkpoints/toto-models/Toto-2.0-2.5B",
        #     "release_date": "2026-05-14",
        # },
    },
}



Model_abbrev_map = {
    f"{family}_{variant}": info["abbreviation"]
    for family, variants in Model_zoo_details.items()
    for variant, info in variants.items()
    if "abbreviation" in info
}

All_model_names = [
    f"{family}_{variant}"
    for family, variants in Model_zoo_details.items()
    for variant, info in variants.items()
    if "abbreviation" in info
]

_all_models_with_date = []
for family, variants in Model_zoo_details.items():
    for variant, info in variants.items():
        if "abbreviation" not in info:
            continue
        model_name = f"{family}_{variant}"
                                  
        release_date = info.get("release_date", "2100-01-01")
        _all_models_with_date.append((release_date, model_name))

All_sorted_model_names = [
    name for _, name in sorted(_all_models_with_date, key=lambda x: x[0])
]


def build_model_family_metadata(model_abbr_order):
    """Build the persisted Step3 family/size contract for one zoo stage."""
    requested = [str(value) for value in model_abbr_order]
    requested_set = set(requested)
    family_by_abbr = {}
    size_variant_by_abbr = {}
    size_rank_by_abbr = {}
    family_members = {}

    for family, variants in Model_zoo_details.items():
        members = []
        for size_rank, (variant, info) in enumerate(variants.items()):
            abbr = str(info.get("abbreviation", ""))
            if not abbr or abbr not in requested_set:
                continue
            family_by_abbr[abbr] = str(family)
            size_variant_by_abbr[abbr] = str(variant)
            size_rank_by_abbr[abbr] = int(size_rank)
            members.append(abbr)
        if members:
            family_members[str(family)] = members

    missing = [abbr for abbr in requested if abbr not in family_by_abbr]
    if missing:
        raise ValueError(f"model family metadata missing abbreviations: {missing}")

    return {
        "model_family_schema_version": 1,
        "model_family_by_abbr": family_by_abbr,
        "model_size_variant_by_abbr": size_variant_by_abbr,
        "model_size_rank_by_abbr": size_rank_by_abbr,
        "model_family_members": family_members,
    }


def validate_model_family_metadata(metadata, expected_model_order):
    """Return validation problems for a persisted Step3 family contract."""
    expected = [str(value) for value in expected_model_order]
    if not isinstance(metadata, dict):
        return ["family metadata is not a mapping"]
    family_by_abbr = metadata.get("model_family_by_abbr")
    family_members = metadata.get("model_family_members")
    size_rank_by_abbr = metadata.get("model_size_rank_by_abbr")
    if not isinstance(family_by_abbr, dict):
        return ["missing model_family_by_abbr"]
    if not isinstance(family_members, dict):
        return ["missing model_family_members"]
    if not isinstance(size_rank_by_abbr, dict):
        return ["missing model_size_rank_by_abbr"]

    problems = []
    expected_set = set(expected)
    if set(str(key) for key in family_by_abbr) != expected_set:
        problems.append("model_family_by_abbr keys do not match model_abbr_order")
    if set(str(key) for key in size_rank_by_abbr) != expected_set:
        problems.append("model_size_rank_by_abbr keys do not match model_abbr_order")

    flattened = []
    for family, raw_members in family_members.items():
        if not isinstance(raw_members, list) or not raw_members:
            problems.append(f"family {family!r} has no ordered members")
            continue
        members = [str(value) for value in raw_members]
        flattened.extend(members)
        for abbr in members:
            if str(family_by_abbr.get(abbr, "")) != str(family):
                problems.append(f"family mismatch for {abbr!r}")
        ranks = [size_rank_by_abbr.get(abbr) for abbr in members]
        try:
            numeric_ranks = [int(value) for value in ranks]
        except (TypeError, ValueError):
            problems.append(f"family {family!r} has invalid size ranks")
        else:
            if numeric_ranks != sorted(numeric_ranks):
                problems.append(f"family {family!r} members are not in ascending size order")

    if len(flattened) != len(set(flattened)):
        problems.append("model_family_members contains duplicate abbreviations")
    if set(flattened) != expected_set:
        problems.append("model_family_members does not cover model_abbr_order")
    return problems


def build_route_family_target_by_model_id(model_abbr_order, metadata, mode):
    """Map every model id to its selected family representative id."""
    order = [str(value) for value in model_abbr_order]
    mode = str(mode or "default").strip().lower()
    if mode == "default":
        return list(range(len(order)))
    if mode not in {"bigger_size", "smaller_size"}:
        raise ValueError(
            f"Unknown route_family_mode={mode!r}; use default, bigger_size, or smaller_size"
        )
    problems = validate_model_family_metadata(metadata, order)
    if problems:
        raise ValueError("invalid model family metadata: " + "; ".join(problems))

    id_by_abbr = {abbr: idx for idx, abbr in enumerate(order)}
    target_by_id = list(range(len(order)))
    for raw_members in metadata["model_family_members"].values():
        members = [str(value) for value in raw_members]
        target_abbr = members[-1] if mode == "bigger_size" else members[0]
        target_id = int(id_by_abbr[target_abbr])
        for abbr in members:
            target_by_id[int(id_by_abbr[abbr])] = target_id
    return target_by_id


MODEL_DEFAULT_BATCH_SIZE = 128

Model_default_batch_sizes = {
    "moirai_small": 1024,
    "moirai_base": 512,
    "moirai_large": 256,
    "chronos_bolt_tiny": 1024,
    "chronos_bolt_mini": 1024,
    "chronos_bolt_small": 512,
    "chronos_bolt_base": 512,
    "moirai2_small": 1024,
    "flowstate_r1": 128,
    "kairos_10m": 1024,
    "kairos_23m": 1024,
    "kairos_50m": 1024,
    "timesfm_2.5": 1024,
    "chronos2_base": 512,
    "patchtst_r1": 512,
    "timemoe_50m": 128,
    "toto_base": 16,
    "tirex_1.1": 512,
    "toto2_4m": 1024,
    "toto2_22m": 1024,
    "toto2_313m": 256,
    "toto2_1B": 128,
    "toto2_2.5B": 64,
}


def get_model_default_batch_size(model_name: str, fallback: int | None = None) -> int | None:
    configured = Model_default_batch_sizes.get(str(model_name))
    if configured is None:
        return fallback
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        return fallback
    return configured if configured > 0 else fallback


MULTIVAR_TSFM_PREFIXES = [
    "moirai",
    "toto",
    "toto2",
    "chronos2",
                                                                     
]
