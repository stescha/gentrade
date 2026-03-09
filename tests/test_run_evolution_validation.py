"""Tests that run_evolution enforces the new data/label input invariants.

This is mostly sanity checking the helper logic added during the refactor;
upstream callers (the public API) still freely accept dicts for convenience but
internal operations always receive ordered lists.
"""

from typing import cast

import pandas as pd
import pytest

from gentrade.config import RunConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.evolve import run_evolution


@pytest.fixture
def cfg() -> RunConfig:
    return RunConfig()


def _make_sample(n: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    df = generate_synthetic_ohlcv(n, seed=0)
    labels = pd.Series([True] * n, index=df.index)
    return df, labels


@pytest.mark.unit
class TestRunEvolutionDataLabelValidation:
    def test_mismatched_train_keys(self, cfg: RunConfig) -> None:
        df, labels = _make_sample()
        df2, labels2 = _make_sample()
        with pytest.raises(ValueError, match="same keys"):
            run_evolution({"a": df}, {"b": labels}, None, None, cfg)

    def test_mismatched_val_keys(self, cfg: RunConfig) -> None:
        df, labels = _make_sample()
        with pytest.raises(ValueError, match="same keys"):
            run_evolution(df, labels, {"x": df}, {"y": labels}, cfg)

    def test_train_index_mismatch(self, cfg: RunConfig) -> None:
        df, labels = _make_sample()
        wrong = labels.copy()
        wrong.index = cast(pd.DatetimeIndex, wrong.index).shift(1)
        with pytest.raises(ValueError, match="Index mismatch"):
            run_evolution({"a": df}, {"a": wrong}, None, None, cfg)

    def test_val_index_mismatch(self, cfg: RunConfig) -> None:
        df, labels = _make_sample()
        wrong = labels.copy()
        wrong.index = cast(pd.DatetimeIndex, wrong.index).shift(1)
        with pytest.raises(ValueError, match="Index mismatch"):
            run_evolution(df, labels, {"v": df}, {"v": wrong}, cfg)

    def test_non_dict_label_type_raises(self, cfg: RunConfig) -> None:
        df, labels = _make_sample()
        # train data is dict but labels provided as Series
        with pytest.raises(ValueError, match="must also be a dict"):
            run_evolution({"a": df}, labels, None, None, cfg)
        # val data is dict but labels provided as Series
        with pytest.raises(ValueError, match="must also be a dict"):
            run_evolution(df, labels, {"v": df}, labels, cfg)
