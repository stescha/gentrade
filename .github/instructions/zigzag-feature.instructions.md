# Zigzag feature

Detects peaks/valleys in a numeric series. Output array has `1` for
peaks, `-1` for valleys, `0` for none. Uses two relative thresholds.

**Look‑ahead bias** – future data are considered. Not predictive;
usable only as a label or post‑hoc tag.

Python API:
```python
peak_valley_pivots(X, up_thresh: float, down_thresh: float) -> np.ndarray
peak_valley_pivots_detailed(
    X: np.ndarray,
    up_thresh: float,
    down_thresh: float,
    limit_to_finalized_segments: bool,
    use_eager_switching_for_non_final: bool,
) -> np.ndarray
```
`_to_ndarray` coerces input. Dependencies: `numpy`, `Cython`. Implementation
is pure computation.