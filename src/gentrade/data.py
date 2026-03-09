import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(n: int, seed: None | int = None) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movements.

    Args:
        n: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    rng = np.random.default_rng(seed)

    returns = rng.normal(0, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))

    noise = rng.uniform(0.001, 0.01, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(1000, 10000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("1990-01-01", periods=n, freq="1min"),
    )


# def prepare_data(cfg: RunConfig) -> pd.DataFrame:
#     """Load or generate OHLCV data according to a run configuration.

#     This function encapsulates the logic that used to live inside
#     :func:`run_evolution`. Calling code may use it to fetch data once and then
#     pass the resulting DataFrame into :func:`run_evolution`.

#     Args:
#         cfg: Run configuration containing ``DataConfig`` parameters.

#     Returns:
#         OHLCV ``DataFrame`` suitable for evolution. For synthetically generated
#         data the random seed from ``cfg`` is reused for reproducibility.
#     """
#     if cfg.data.pair is not None:
#         from gentrade.tradetools import load_binance_ohlcv

#         df = load_binance_ohlcv(
#             cfg.data.pair,
#             cfg.data.start,
#             cfg.data.stop,
#             cfg.data.count,
#         )
#         print(f"Loaded real OHLCV data for {cfg.data.pair}: {len(df)} rows")
#     else:
#         df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
#         print(f"Generated synthetic OHLCV data: {len(df)} rows")
#     return df
