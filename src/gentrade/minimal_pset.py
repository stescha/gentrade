"""Minimal pset factory for zigzag GP smoke testing.

Provides composable functions to build typed primitive sets with zigzag
and TA-Lib indicators.
"""

import operator

import pandas as pd
import talib
from deap import gp
from zigzag import peak_valley_pivots  # type: ignore[import-untyped]

from gentrade.pset.pset_types import (
    Acceleration,
    BooleanSeries,
    Close,
    FastLimit,
    High,
    Label,
    Low,
    MAType,
    Maximum,
    NBDev,
    NumericSeries,
    Open,
    PriceSeries,
    SlowLimit,
    Threshold,
    Timeperiod,
    VFactor,
    Volume,
)
from gentrade.pset.talib_primitives import (
    BBANDS_lowerband,
    BBANDS_middleband,
    BBANDS_upperband,
    MACD_macd,
    MACD_macdsignal,
    STOCH_slowd,
    STOCH_slowk,
    add_cycle_indicators,
    add_momentum_indicators,
    add_overlap_studies,
    add_statistic_functions,
    add_volatility_indicators,
    add_volume_indicators,
)


def _calc_zigzag_pivots(
    price: pd.Series,
    threshold: float,
    label: int,
) -> pd.Series:
    pivots = peak_valley_pivots(price.values, threshold, -threshold)
    return pd.Series(pivots == label, index=price.index)


def zigzag_pivots(
    data: pd.Series | pd.DataFrame | dict[str, pd.DataFrame],
    threshold: float,
    label: int,
    column: str = "close",
) -> pd.Series | dict[str, pd.Series]:
    """Compute zigzag pivots and return boolean mask where pivot == label.

    Uses look-ahead — intentionally a 'cheat' primitive for smoke testing.
    """
    if isinstance(data, pd.Series):
        return _calc_zigzag_pivots(data, threshold, label)
    elif isinstance(data, pd.DataFrame):
        return _calc_zigzag_pivots(data[column], threshold, label)
    elif isinstance(data, dict):
        return {
            key: _calc_zigzag_pivots(df[column], threshold, label)
            for key, df in data.items()
        }


def not_(s: pd.Series) -> pd.Series:
    """Boolean NOT for series."""
    return ~s


def create_pset_core(name: str = "core") -> gp.PrimitiveSetTyped:
    """Create base pset with OHLCV inputs, logical/comparison ops, and terminals.

    Args:
        name: Name for the primitive set.

    Returns:
        A typed primitive set with core operators registered.
    """
    pset = gp.PrimitiveSetTyped(name, [Open, High, Low, Close, Volume], BooleanSeries)

    # Logical operators
    pset.addPrimitive(operator.and_, [BooleanSeries, BooleanSeries], BooleanSeries)
    pset.addPrimitive(operator.or_, [BooleanSeries, BooleanSeries], BooleanSeries)
    pset.addPrimitive(not_, [BooleanSeries], BooleanSeries, name="not_")

    # Comparison operators
    for op in [operator.gt, operator.lt, operator.ge, operator.le]:
        pset.addPrimitive(op, [NumericSeries, NumericSeries], BooleanSeries)

    # Boolean terminals
    pset.addTerminal(True, BooleanSeries)
    pset.addTerminal(False, BooleanSeries)

    # Ephemeral constants for zigzag
    pset.addEphemeralConstant("Threshold", Threshold.sample, Threshold)
    pset.addEphemeralConstant("Label", Label.sample, Label)

    # Rename arguments
    pset.renameArguments(
        ARG0="open", ARG1="high", ARG2="low", ARG3="close", ARG4="volume"
    )

    return pset


def add_zigzag_cheat(pset: gp.PrimitiveSetTyped) -> None:
    """Register the zigzag_pivots cheat primitive on an existing pset."""
    pset.addPrimitive(zigzag_pivots, [Close, Threshold, Label], BooleanSeries)


def add_features_minimal(pset: gp.PrimitiveSetTyped) -> None:
    """Add ~8 strongest core TA-Lib indicator primitives.

    Includes RSI, SMA, EMA, ATR, Bollinger Bands, and ADX.
    """
    # Single-output indicators
    pset.addPrimitive(talib.RSI, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.SMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.EMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ATR, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ADX, [High, Low, Close, Timeperiod], NumericSeries)

    # Bollinger Bands (multi-output split)
    pset.addPrimitive(
        BBANDS_upperband, [PriceSeries, Timeperiod, NBDev, NBDev, MAType], NumericSeries
    )
    pset.addPrimitive(
        BBANDS_middleband,
        [PriceSeries, Timeperiod, NBDev, NBDev, MAType],
        NumericSeries,
    )
    pset.addPrimitive(
        BBANDS_lowerband, [PriceSeries, Timeperiod, NBDev, NBDev, MAType], NumericSeries
    )

    # Ephemeral constants for these primitives
    pset.addEphemeralConstant("Timeperiod", Timeperiod.sample, Timeperiod)
    pset.addEphemeralConstant("NBDev", NBDev.sample, NBDev)
    pset.addEphemeralConstant("MAType", MAType.sample, MAType)


def add_features_medium(pset: gp.PrimitiveSetTyped) -> None:
    """Add ~20 additional TA-Lib indicator primitives beyond minimal.

    Calls add_features_minimal first, then adds MACD, CCI, Stochastic, etc.
    """
    add_features_minimal(pset)

    # MACD
    pset.addPrimitive(
        MACD_macd, [PriceSeries, Timeperiod, Timeperiod, Timeperiod], NumericSeries
    )
    pset.addPrimitive(
        MACD_macdsignal,
        [PriceSeries, Timeperiod, Timeperiod, Timeperiod],
        NumericSeries,
    )

    # CCI
    pset.addPrimitive(talib.CCI, [High, Low, Close, Timeperiod], NumericSeries)

    # Stochastic
    pset.addPrimitive(
        STOCH_slowk,
        [High, Low, Close, Timeperiod, Timeperiod, MAType, Timeperiod, MAType],
        NumericSeries,
    )
    pset.addPrimitive(
        STOCH_slowd,
        [High, Low, Close, Timeperiod, Timeperiod, MAType, Timeperiod, MAType],
        NumericSeries,
    )

    # Momentum indicators
    pset.addPrimitive(talib.MOM, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ROC, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.WILLR, [High, Low, Close, Timeperiod], NumericSeries)

    # Volume
    pset.addPrimitive(talib.OBV, [PriceSeries, Volume], NumericSeries)

    # Volatility
    pset.addPrimitive(talib.NATR, [High, Low, Close, Timeperiod], NumericSeries)

    # More overlap studies
    pset.addPrimitive(talib.DEMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.KAMA, [PriceSeries, Timeperiod], NumericSeries)

    # Statistics
    pset.addPrimitive(talib.LINEARREG_SLOPE, [PriceSeries, Timeperiod], NumericSeries)

    # Money Flow Index
    pset.addPrimitive(talib.MFI, [High, Low, Close, Volume, Timeperiod], NumericSeries)

    # Directional indicators
    pset.addPrimitive(talib.MINUS_DI, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.PLUS_DI, [High, Low, Close, Timeperiod], NumericSeries)

    # TRIX
    pset.addPrimitive(talib.TRIX, [PriceSeries, Timeperiod], NumericSeries)


def add_features_large(pset: gp.PrimitiveSetTyped) -> None:
    """Add all available TA-Lib indicators.

    Calls add_features_medium first, then adds remaining indicators via
    add_talib_indicators, handling duplicates gracefully.
    """
    add_features_medium(pset)

    # Additional ephemeral constants needed by full talib set
    pset.addEphemeralConstant("slowlimit", SlowLimit.sample, SlowLimit)
    pset.addEphemeralConstant("fastlimit", FastLimit.sample, FastLimit)
    pset.addEphemeralConstant("vfactor", VFactor.sample, VFactor)
    pset.addEphemeralConstant("acceleration", Acceleration.sample, Acceleration)
    pset.addEphemeralConstant("maximum", Maximum.sample, Maximum)

    # Wrap each category to handle duplicate primitive errors
    # TODO:
    # Temporarily allow type ignores because `talib_primitives` is untyped.
    # Remove aftere pset rework.
    add_cycle_indicators(pset)  # type: ignore
    add_momentum_indicators(pset)  # type: ignore
    add_overlap_studies(pset)  # type: ignore
    add_statistic_functions(pset)  # type: ignore
    add_volatility_indicators(pset)  # type: ignore
    add_volume_indicators(pset)  # type: ignore


def create_pset_zigzag_minimal(name: str = "zigzag_minimal") -> gp.PrimitiveSetTyped:
    """Create pset with zigzag cheat and minimal (~8) TA-Lib features."""
    pset = create_pset_core(name)
    add_zigzag_cheat(pset)
    add_features_minimal(pset)
    return pset


def create_pset_zigzag_medium(name: str = "zigzag_medium") -> gp.PrimitiveSetTyped:
    """Create pset with zigzag cheat and medium (~20) TA-Lib features."""
    pset = create_pset_core(name)
    add_zigzag_cheat(pset)
    add_features_medium(pset)
    return pset


def create_pset_zigzag_large(name: str = "zigzag_large") -> gp.PrimitiveSetTyped:
    """Create pset with zigzag cheat and all available TA-Lib features."""
    pset = create_pset_core(name)
    add_zigzag_cheat(pset)
    add_features_large(pset)
    return pset


def create_pset_default_medium(name: str = "default_medium") -> gp.PrimitiveSetTyped:
    """Create medium sized pset with default features."""
    pset = create_pset_core(name)
    add_features_medium(pset)
    return pset


def create_pset_default_large(name: str = "default_large") -> gp.PrimitiveSetTyped:
    """Create large pset with default features."""
    pset = create_pset_core(name)
    add_features_large(pset)
    return pset
