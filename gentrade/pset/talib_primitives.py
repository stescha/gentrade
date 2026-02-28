import talib
from gentrade.pset.pset_types import (
    BooleanSeries,
    Close,
    FastLimit,
    High,
    Low,
    MAType,
    Maximum,
    NBDev,
    NumericSeries,
    Open,
    PriceSeries,
    SlowLimit,
    Timeperiod,
    VFactor,
    Volume,
)


def HT_PHASOR_inphase(close):
    return talib.HT_PHASOR(close)[0]


def HT_PHASOR_quadrature(close):
    return talib.HT_PHASOR(close)[1]


def HT_SINE_sine(close):
    return talib.HT_SINE(close)[0]


def HT_SINE_leadsine(close):
    return talib.HT_SINE(close)[1]


def AROON_aroondown(high, low, timeperiod):
    return talib.AROON(high, low, timeperiod)[0]


def AROON_aroonup(high, low, timeperiod):
    return talib.AROON(high, low, timeperiod)[1]


def MACD_macd(close, fastperiod, slowperiod, signalperiod):
    return talib.MACD(close, fastperiod, slowperiod, signalperiod)[0]


def MACD_macdsignal(close, fastperiod, slowperiod, signalperiod):
    return talib.MACD(close, fastperiod, slowperiod, signalperiod)[1]


def MACDEXT_macd(close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype):
    return talib.MACDEXT(close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)[0]


def MACDEXT_macdsignal(close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype):
    return talib.MACDEXT(close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)[1]


def MACDFIX_macd(close, signalperiod):
    return talib.MACDFIX(close, signalperiod)[0]


def MACDFIX_macdsignal(close, signalperiod):
    return talib.MACDFIX(close, signalperiod)[1]


def STOCH_slowk(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype):
    return talib.STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)[0]


def STOCH_slowd(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype):
    return talib.STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)[1]


def STOCHF_fastk(high, low, close, fastk_period, fastd_period, fastd_matype):
    return talib.STOCHF(high, low, close, fastk_period, fastd_period, fastd_matype)[0]


def STOCHF_fastd(high, low, close, fastk_period, fastd_period, fastd_matype):
    return talib.STOCHF(high, low, close, fastk_period, fastd_period, fastd_matype)[1]


def STOCHRSI_fastk(close, timeperiod, fastk_period, fastd_period, fastd_matype):
    return talib.STOCHRSI(close, timeperiod, fastk_period, fastd_period, fastd_matype)[0]


def STOCHRSI_fastd(close, timeperiod, fastk_period, fastd_period, fastd_matype):
    return talib.STOCHRSI(close, timeperiod, fastk_period, fastd_period, fastd_matype)[1]


def BBANDS_upperband(close, timeperiod, nbdevup, nbdevdn, matype):
    return talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)[0]


def BBANDS_middleband(close, timeperiod, nbdevup, nbdevdn, matype):
    return talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)[1]


def BBANDS_lowerband(close, timeperiod, nbdevup, nbdevdn, matype):
    return talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)[2]


def MAMA_mama(close, fastlimit, slowlimit):
    return talib.MAMA(close, fastlimit, slowlimit)[0]


def MAMA_fama(close, fastlimit, slowlimit):
    return talib.MAMA(close, fastlimit, slowlimit)[1]


def add_cycle_indicators(pset):
    """Add cycle indicator primitives to a DEAP primitive set."""
    pset.addPrimitive(talib.HT_DCPERIOD, [PriceSeries], NumericSeries)
    pset.addPrimitive(talib.HT_DCPHASE, [PriceSeries], NumericSeries)
    pset.addPrimitive(HT_PHASOR_inphase, [PriceSeries], NumericSeries)
    pset.addPrimitive(HT_PHASOR_quadrature, [PriceSeries], NumericSeries)
    pset.addPrimitive(HT_SINE_sine, [PriceSeries], NumericSeries)
    pset.addPrimitive(HT_SINE_leadsine, [PriceSeries], NumericSeries)


def add_momentum_indicators(pset):
    """Add momentum indicator primitives to a DEAP primitive set."""
    pset.addPrimitive(talib.ADX, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ADXR, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.APO, [PriceSeries, Timeperiod, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(AROON_aroondown, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(AROON_aroonup, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.AROONOSC, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.BOP, [Open, High, Low, Close], NumericSeries)
    pset.addPrimitive(talib.CCI, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.CMO, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.DX, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(MACD_macd, [PriceSeries, Timeperiod, Timeperiod, Timeperiod], NumericSeries)
    pset.addPrimitive(MACD_macdsignal, [PriceSeries, Timeperiod, Timeperiod, Timeperiod], NumericSeries)
    pset.addPrimitive(MACDEXT_macd, [PriceSeries, Timeperiod, MAType, Timeperiod, MAType, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(MACDEXT_macdsignal, [PriceSeries, Timeperiod, MAType, Timeperiod, MAType, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(MACDFIX_macd, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(MACDFIX_macdsignal, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.MFI, [High, Low, Close, Volume, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.MINUS_DI, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.MINUS_DM, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.MOM, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.PLUS_DI, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.PLUS_DM, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.PPO, [PriceSeries, Timeperiod, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(talib.ROC, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ROCP, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ROCR, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ROCR100, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.RSI, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(STOCH_slowk, [High, Low, Close, Timeperiod, Timeperiod, MAType, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(STOCH_slowd, [High, Low, Close, Timeperiod, Timeperiod, MAType, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(STOCHF_fastk, [High, Low, Close, Timeperiod, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(STOCHF_fastd, [High, Low, Close, Timeperiod, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(STOCHRSI_fastk, [PriceSeries, Timeperiod, Timeperiod, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(STOCHRSI_fastd, [PriceSeries, Timeperiod, Timeperiod, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(talib.TRIX, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.ULTOSC, [High, Low, Close, Timeperiod, Timeperiod, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.WILLR, [High, Low, Close, Timeperiod], NumericSeries)


def add_overlap_studies(pset):
    """Add overlap study primitives to a DEAP primitive set."""
    pset.addPrimitive(BBANDS_upperband, [PriceSeries, Timeperiod, NBDev, NBDev, MAType], NumericSeries)
    pset.addPrimitive(BBANDS_middleband, [PriceSeries, Timeperiod, NBDev, NBDev, MAType], NumericSeries)
    pset.addPrimitive(BBANDS_lowerband, [PriceSeries, Timeperiod, NBDev, NBDev, MAType], NumericSeries)
    pset.addPrimitive(talib.DEMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.EMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.HT_TRENDLINE, [PriceSeries], NumericSeries)
    pset.addPrimitive(talib.KAMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.MA, [PriceSeries, Timeperiod, MAType], NumericSeries)
    pset.addPrimitive(MAMA_mama, [PriceSeries, FastLimit, SlowLimit], NumericSeries)
    pset.addPrimitive(MAMA_fama, [PriceSeries, FastLimit, SlowLimit], NumericSeries)
    pset.addPrimitive(talib.MIDPOINT, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.MIDPRICE, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.SAR, [High, Low], NumericSeries)
    pset.addPrimitive(talib.SMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.T3, [PriceSeries, Timeperiod, VFactor], NumericSeries)
    pset.addPrimitive(talib.TEMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.TRIMA, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.WMA, [PriceSeries, Timeperiod], NumericSeries)


def add_statistic_functions(pset):
    """Add statistic function primitives to a DEAP primitive set."""
    pset.addPrimitive(talib.BETA, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.CORREL, [High, Low, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.LINEARREG, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.LINEARREG_ANGLE, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.LINEARREG_INTERCEPT, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.LINEARREG_SLOPE, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.STDDEV, [PriceSeries, Timeperiod, NBDev], NumericSeries)
    pset.addPrimitive(talib.TSF, [PriceSeries, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.VAR, [PriceSeries, Timeperiod, NBDev], NumericSeries)


def add_volatility_indicators(pset):
    """Add volatility indicator primitives to a DEAP primitive set."""
    pset.addPrimitive(talib.ATR, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.NATR, [High, Low, Close, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.TRANGE, [High, Low, Close], NumericSeries)


def add_volume_indicators(pset):
    """Add volume indicator primitives to a DEAP primitive set."""
    pset.addPrimitive(talib.AD, [High, Low, Close, Volume], NumericSeries)
    pset.addPrimitive(talib.ADOSC, [High, Low, Close, Volume, Timeperiod, Timeperiod], NumericSeries)
    pset.addPrimitive(talib.OBV, [PriceSeries, Volume], NumericSeries)


def add_ephemeral_constants(pset):
    """Add ephemeral constants for TA-Lib parameters to a primitive set."""
    from gentrade.pset.pset_types import (
        Acceleration,
        FastLimit,
        MAType,
        Maximum,
        NBDev,
        SlowLimit,
        Timeperiod,
        VFactor,
    )
    pset.addEphemeralConstant("slowlimit", SlowLimit.sample, SlowLimit)
    pset.addEphemeralConstant("nbdev", NBDev.sample, NBDev)
    pset.addEphemeralConstant("vfactor", VFactor.sample, VFactor)
    pset.addEphemeralConstant("acceleration", Acceleration.sample, Acceleration)
    pset.addEphemeralConstant("fastlimit", FastLimit.sample, FastLimit)
    pset.addEphemeralConstant("matype", MAType.sample, MAType)
    pset.addEphemeralConstant("timeperiod", Timeperiod.sample, Timeperiod)
    pset.addEphemeralConstant("maximum", Maximum.sample, Maximum)


def add_talib_indicators(pset):
    """Add all TA-Lib indicator primitives and ephemeral constants to a pset."""
    add_cycle_indicators(pset)
    add_momentum_indicators(pset)
    add_overlap_studies(pset)
    add_statistic_functions(pset)
    add_volatility_indicators(pset)
    add_volume_indicators(pset)
    add_ephemeral_constants(pset)
