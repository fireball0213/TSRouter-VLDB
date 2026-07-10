# utils/missing.py
import numpy as np
from scipy.interpolate import interp1d

def fill_missing_1d(
    x: np.ndarray,
    *,
    all_nan_strategy: str = "zero",                           
    interp_kind: str = "linear",                              
    add_noise: bool = False,                           
    noise_ratio: float = 0.01,                       
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

                                     
    finite_mask = np.isfinite(x)

                       
    if not finite_mask.any():
        if all_nan_strategy == "linspace":
            return np.linspace(0.0, 1.0, len(x), dtype=float)
        else:                    
            return np.zeros_like(x, dtype=float)

               
    if finite_mask.all():
        filled = x.copy()
    else:
        idx = np.arange(len(x))
        known_idx = idx[finite_mask]
        known_vals = x[finite_mask]

                           
        if len(known_idx) == 1:
            filled = np.full_like(x, known_vals[0], dtype=float)
        else:
            if interp_kind == "nearest":
                kind = "nearest"
            else:
                kind = "linear"
            f = interp1d(
                known_idx,
                known_vals,
                kind=kind,
                fill_value="extrapolate",
            )
            filled = f(idx).astype(float)

                                     
    if add_noise:
        base = float(np.mean(np.abs(filled))) if np.isfinite(filled).any() else 1.0
        scale = noise_ratio * max(base, 1e-8)
        noise = np.random.normal(0.0, scale, size=len(filled))
        filled = filled + noise

    return filled


def fill_missing(
    arr: np.ndarray,
    *,
    all_nan_strategy_1d: str = "zero",
    all_nan_strategy_2d_global: str | None = None,
    interp_kind_1d: str = "linear",
    interp_kind_2d: str = "linear",
    add_noise_1d: bool = False,
    noise_ratio_1d: float = 0.01,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return fill_missing_1d(
            arr,
            all_nan_strategy=all_nan_strategy_1d,
            interp_kind=interp_kind_1d,
            add_noise=add_noise_1d,
            noise_ratio=noise_ratio_1d,
        )

    if arr.ndim == 2:
                            
        if all_nan_strategy_2d_global == "linspace" and np.isnan(arr).all():
            T = arr.shape[1]
            base = np.linspace(0.0, 1.0, T, dtype=float)
            return np.vstack([base.copy() for _ in range(arr.shape[0])])

        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            out[i] = fill_missing_1d(
                arr[i],
                                                          
                all_nan_strategy="zero",
                interp_kind=interp_kind_2d,
                add_noise=False,
            )
        return out

    raise ValueError(f"TSRouter runtime message: {arr.ndim}")
