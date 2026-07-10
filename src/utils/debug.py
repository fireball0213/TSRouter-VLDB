import copy
import numpy as np
import itertools

def debug_check_input_nan(test_data_input):
    for i, entry in enumerate(test_data_input):
        tgt = np.asarray(entry["target"], dtype=float)
        if not np.all(np.isfinite(tgt)):
            n_nan = np.isnan(tgt).sum()
            n_posinf = np.isposinf(tgt).sum()
            n_neginf = np.isneginf(tgt).sum()
            print(
                f"[DATA NaN DEBUG] series {i} has non-finite values: "
                f"NaN={n_nan}, +Inf={n_posinf}, -Inf={n_neginf}, shape={tgt.shape}"
            )
                            
            bad_idx = np.where(~np.isfinite(tgt))[0][:10]
            print("  bad indices (first 10):", bad_idx)
            break         


def debug_print_test_input(dataset, max_items: int = 1):
    """Inspect dataset.test_data.input for debugging."""
    test_input = dataset.test_data.input

    print("[DEBUG] type(dataset.test_data.input):", type(test_input))
    try:
        print("[DEBUG] len(dataset.test_data.input):", len(test_input))
    except TypeError:
        print("[DEBUG] len(...) not supported for this type")

                       
    if isinstance(test_input, (list, tuple)):
        samples = list(test_input[:max_items])
    else:
        samples = list(itertools.islice(test_input, max_items))

    for idx, item in enumerate(samples):
        print(f"\n[DEBUG] Sample #{idx} type:", type(item))
        if isinstance(item, dict):
            for k, v in item.items():
                if hasattr(v, "shape"):
                    print(f"  key={k}, type={type(v)}, shape={v.shape}, "
                          f"dtype={getattr(v, 'dtype', None)}")
                else:
                                       
                    if isinstance(v, (int, float, str)):
                        print(f"  key={k}, type={type(v)}, value={v}")
                    else:
                        print(f"  key={k}, type={type(v)}, value=...")
        else:
            if hasattr(item, "shape"):
                print(f"  shape={item.shape}, dtype={getattr(item, 'dtype', None)}")
            else:
                print("  value:", item)


def debug_forecasts(forecasts):
    if len(forecasts) == 0:
        print('TSRouter runtime message.')
        raise ValueError('TSRouter runtime message.')

                                                            
    fc0 = forecasts[0]
    if hasattr(fc0, "samples"):
        print(f"SampleForecasts shape: {len(forecasts)} * ", fc0.samples.shape)
    elif hasattr(fc0, "forecast_array"):
        print(
            f"QuantileForecasts shape: {len(forecasts)} * ",
            fc0.forecast_array.shape,
        )
    else:
        print(f"TSRouter runtime message: {len(forecasts)}")
        print(f"TSRouter runtime message: {type(fc0)}")
        if hasattr(fc0, "__dict__"):
            print(f"TSRouter runtime message: ")
            for attr_name, attr_value in fc0.__dict__.items():
                print(
                    f"  - {attr_name}TSRouter runtime message: {type(attr_value)}, "
                    f"TSRouter runtime message: {str(attr_value)[:200]}"
                )
        raise ValueError(
            "forecasts[0] does not have 'samples' or 'forecast_array' attribute"
        )

def debug_dataset_brief( dataset, tag: str, n: int = 1):
    print(f"\n[DBG][{tag}] dataset type: {type(dataset)}")
    print(f"[DBG][{tag}] dataset.name={getattr(dataset, 'name', None)}  freq={getattr(dataset, 'freq', None)}")
    print(f"[DBG][{tag}] target_dim={getattr(dataset, 'target_dim', None)} windows={getattr(dataset, 'windows', None)}")
    print(f"[DBG][{tag}] prediction_length={getattr(dataset, 'prediction_length', None)}")

    td = getattr(dataset, "test_data", None)
    if td is None:
        print(f"[DBG][{tag}] dataset.test_data is None")
        return

    inp = getattr(td, "input", None)
    lab = getattr(td, "label", None)
    print(f"[DBG][{tag}] test_data.input type={type(inp)} len={len(inp) if inp is not None else None}")
    print(f"[DBG][{tag}] test_data.label type={type(lab)} len={len(lab) if lab is not None else None}")

                       
    if inp is not None and len(inp) > 0:
        x0 = inp[0]
        if isinstance(x0, dict) and "target" in x0:
            print(f"[DBG][{tag}] input[0].target shape={np.asarray(x0['target']).shape} dtype={np.asarray(x0['target']).dtype}")
        print(f"[DBG][{tag}] input[0].start type={type(x0.get('start', None))} value={x0.get('start', None)}")
        print(f"[DBG][{tag}] input[0].freq={x0.get('freq', None)}")

    if lab is not None and len(lab) > 0:
        y0 = lab[0]
        if isinstance(y0, dict) and "target" in y0:
            print(f"[DBG][{tag}] label[0].target shape={np.asarray(y0['target']).shape} dtype={np.asarray(y0['target']).dtype}")
        print(f"[DBG][{tag}] label[0].start type={type(y0.get('start', None))} value={y0.get('start', None)}")
        print(f"[DBG][{tag}] label[0].freq={y0.get('freq', None)}")


def debug_predictor_brief(predictor, tag: str):
    print(f"\n[DBG][{tag}] predictor type: {type(predictor)}")
    for k in ["prediction_length", "context_length", "freq", "device"]:
        if hasattr(predictor, k):
            print(f"[DBG][{tag}] predictor.{k} = {getattr(predictor, k)}")


def debug_forecast_brief(forecasts, tag: str, n: int = 1):
    if forecasts is None:
        print(f"[DBG][{tag}] forecasts is None")
        return
    print(f"\n[DBG][{tag}] num_forecasts={len(forecasts)}  forecast[0] type={type(forecasts[0])}")
    if len(forecasts) == 0:
        return
    f0 = forecasts[0]
            
    if hasattr(f0, "samples"):
        arr = np.asarray(f0.samples)
        print(f"[DBG][{tag}] forecast[0].samples shape={arr.shape} dtype={arr.dtype}")
    if hasattr(f0, "forecast_array"):
        arr = np.asarray(f0.forecast_array)
        print(f"[DBG][{tag}] forecast[0].forecast_array shape={arr.shape} dtype={arr.dtype}")
    if hasattr(f0, "item_id"):
        print(f"[DBG][{tag}] forecast[0].item_id={f0.item_id}")
    if hasattr(f0, "start_date"):
        print(f"[DBG][{tag}] forecast[0].start_date={f0.start_date}")
