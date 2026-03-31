"""Smoke tests for the zigzag feature.

Tests peak/valley detection via :func:`~gentrade.minimal_pset.zigzag_pivots`
and :func:`~gentrade.minimal_pset.zigzag_pivots_detailed`, pset factory
functions that include the zigzag cheat primitive, and integration of
zigzag-derived labels with :class:`~gentrade.optimizer.TreeOptimizer`.
"""

import numpy as np
import pandas as pd
import pytest
from deap import gp

from gentrade.classification_metrics import F1Metric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import (
    create_pset_zigzag_large,
    create_pset_zigzag_medium,
    create_pset_zigzag_minimal,
    zigzag_pivots,
    zigzag_pivots_detailed,
)
from gentrade.optimizer import TreeOptimizer
from gentrade.pset.pset_types import Label, Threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def close_series() -> pd.Series:
    """Synthetic close price series."""
    return generate_synthetic_ohlcv(500, 42)["close"]


@pytest.fixture(scope="module")
def ohlcv_df() -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    return generate_synthetic_ohlcv(500, 42)


@pytest.fixture(scope="module")
def oscillating_series() -> np.ndarray:
    """Clearly oscillating series to guarantee both peaks and valleys."""
    t = np.linspace(0, 10 * np.pi, 500)
    return 100.0 + np.sin(t)


# ---------------------------------------------------------------------------
# Unit tests: peak_valley_pivots
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPeakValleyPivots:
    """Unit tests for the raw `peak_valley_pivots` output contract."""

    def test_output_values_are_subset_of_valid_labels(
        self, oscillating_series: np.ndarray
    ) -> None:
        """peak_valley_pivots output contains only {-1, 0, 1}."""
        from zigzag import peak_valley_pivots  # type: ignore

        result = peak_valley_pivots(oscillating_series, 0.05, -0.05)  # type: ignore
        unique_vals = set(result)  # type: ignore
        assert unique_vals.issubset({-1, 0, 1})

    def test_output_length_matches_input(
        self, oscillating_series: np.ndarray
    ) -> None:
        """peak_valley_pivots returns an array of the same length as the input."""
        from zigzag import peak_valley_pivots  # type: ignore

        result = peak_valley_pivots(oscillating_series, 0.05, -0.05)  # type: ignore
        assert len(result) == len(oscillating_series)  # type: ignore

    def test_peaks_and_valleys_both_present(
        self, oscillating_series: np.ndarray
    ) -> None:
        """A clearly oscillating series produces both peaks and valleys."""
        from zigzag import peak_valley_pivots  # type: ignore

        result = peak_valley_pivots(oscillating_series, 0.01, -0.01)  # type: ignore
        assert np.any(result == 1), "Expected peaks in oscillating series"  # type: ignore
        assert np.any(result == -1), "Expected valleys in oscillating series"  # type: ignore


# ---------------------------------------------------------------------------
# Unit tests: peak_valley_pivots_detailed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPeakValleyPivotsDetailed:
    """Unit tests for the `peak_valley_pivots_detailed` output contract."""

    def test_output_values_are_subset_of_valid_labels(
        self, oscillating_series: np.ndarray
    ) -> None:
        """peak_valley_pivots_detailed output contains only {-1, 0, 1}."""
        from zigzag import peak_valley_pivots_detailed  # type: ignore

        result = peak_valley_pivots_detailed(  # type: ignore
            oscillating_series, 0.05, -0.05, True, True
        )
        unique_vals = set(result)  # type: ignore
        assert unique_vals.issubset({-1, 0, 1})

    def test_output_length_matches_input(
        self, oscillating_series: np.ndarray
    ) -> None:
        """peak_valley_pivots_detailed returns an array of the same length as the input."""
        from zigzag import peak_valley_pivots_detailed  # type: ignore

        result = peak_valley_pivots_detailed(  # type: ignore
            oscillating_series, 0.05, -0.05, False, False
        )
        assert len(result) == len(oscillating_series)  # type: ignore

    def test_finalized_segments_flag_does_not_change_length(
        self, oscillating_series: np.ndarray
    ) -> None:
        """Both values of limit_to_finalized_segments return same-length arrays."""
        from zigzag import peak_valley_pivots_detailed  # type: ignore

        r_final = peak_valley_pivots_detailed(  # type: ignore
            oscillating_series, 0.05, -0.05, True, True
        )
        r_not_final = peak_valley_pivots_detailed(  # type: ignore
            oscillating_series, 0.05, -0.05, False, True
        )
        assert len(r_final) == len(r_not_final)  # type: ignore


# ---------------------------------------------------------------------------
# Unit tests: zigzag_pivots
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestZigzagPivots:
    """Unit tests for `zigzag_pivots` with different input types."""

    def test_series_input_returns_boolean_series(
        self, close_series: pd.Series
    ) -> None:
        """zigzag_pivots with a Series input returns a boolean pd.Series."""
        result = zigzag_pivots(close_series, 0.05, 1)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_series_input_preserves_index(self, close_series: pd.Series) -> None:
        """zigzag_pivots preserves the index from the input Series."""
        result = zigzag_pivots(close_series, 0.05, 1)
        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, close_series.index)

    def test_series_input_length_matches(self, close_series: pd.Series) -> None:
        """Output length equals input length."""
        result = zigzag_pivots(close_series, 0.05, 1)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close_series)

    def test_dataframe_input_uses_close_column(self, ohlcv_df: pd.DataFrame) -> None:
        """DataFrame input produces same result as passing the close column directly."""
        result_df = zigzag_pivots(ohlcv_df, 0.05, 1)
        result_series = zigzag_pivots(ohlcv_df["close"], 0.05, 1)
        assert isinstance(result_df, pd.Series)
        assert isinstance(result_series, pd.Series)
        pd.testing.assert_series_equal(result_df, result_series)

    def test_dict_input_returns_dict_of_series(self, ohlcv_df: pd.DataFrame) -> None:
        """Dict input returns a dict mapping each key to a boolean pd.Series."""
        data = {"asset1": ohlcv_df, "asset2": ohlcv_df.copy()}
        result = zigzag_pivots(data, 0.05, 1)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"asset1", "asset2"}
        for val in result.values():
            assert isinstance(val, pd.Series)
            assert val.dtype == bool

    def test_label_1_and_minus_1_do_not_overlap(
        self, close_series: pd.Series
    ) -> None:
        """Peaks (label=1) and valleys (label=-1) never occur at the same index."""
        peaks = zigzag_pivots(close_series, 0.05, 1)
        valleys = zigzag_pivots(close_series, 0.05, -1)
        assert isinstance(peaks, pd.Series)
        assert isinstance(valleys, pd.Series)
        assert not (peaks & valleys).any()


# ---------------------------------------------------------------------------
# Unit tests: zigzag_pivots_detailed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestZigzagPivotsDetailed:
    """Unit tests for `zigzag_pivots_detailed` with different input types."""

    def test_series_input_returns_boolean_series(
        self, close_series: pd.Series
    ) -> None:
        """zigzag_pivots_detailed with a Series returns a boolean pd.Series."""
        result = zigzag_pivots_detailed(close_series, 0.05, -0.05, True, True, 1)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_series_input_length_matches(self, close_series: pd.Series) -> None:
        """Output length equals input length."""
        result = zigzag_pivots_detailed(close_series, 0.05, -0.05, False, False, -1)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close_series)

    def test_series_input_preserves_index(self, close_series: pd.Series) -> None:
        """Index is preserved from the input Series."""
        result = zigzag_pivots_detailed(close_series, 0.05, -0.05, True, False, 1)
        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, close_series.index)

    def test_dataframe_input_uses_close_column(self, ohlcv_df: pd.DataFrame) -> None:
        """DataFrame input produces same result as the close column Series."""
        result_df = zigzag_pivots_detailed(ohlcv_df, 0.05, -0.05, True, True, 1)
        result_series = zigzag_pivots_detailed(
            ohlcv_df["close"], 0.05, -0.05, True, True, 1
        )
        assert isinstance(result_df, pd.Series)
        assert isinstance(result_series, pd.Series)
        pd.testing.assert_series_equal(result_df, result_series)

    def test_dict_input_returns_dict_of_series(self, ohlcv_df: pd.DataFrame) -> None:
        """Dict input returns a dict mapping each key to a boolean pd.Series."""
        data = {"a": ohlcv_df, "b": ohlcv_df.copy()}
        result = zigzag_pivots_detailed(data, 0.05, -0.05, True, True, -1)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        for val in result.values():
            assert isinstance(val, pd.Series)
            assert val.dtype == bool


# ---------------------------------------------------------------------------
# Unit tests: Threshold and Label ephemeral constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEphemeralConstants:
    """Unit tests for `Threshold` and `Label` ephemeral constant samplers."""

    def test_threshold_sample_returns_float(self) -> None:
        """Threshold.sample() returns a float."""
        assert isinstance(Threshold.sample(), float)

    def test_threshold_sample_in_range(self) -> None:
        """Threshold.sample() always returns a value in [0.01, 0.10]."""
        for _ in range(100):
            val = Threshold.sample()
            assert 0.01 <= val <= 0.10, f"Threshold {val!r} out of range [0.01, 0.10]"

    def test_label_sample_returns_int(self) -> None:
        """Label.sample() returns an int."""
        assert isinstance(Label.sample(), int)

    def test_label_sample_returns_only_valid_values(self) -> None:
        """Label.sample() returns -1 or 1 only."""
        samples = {Label.sample() for _ in range(100)}
        assert samples.issubset({-1, 1})

    def test_label_sample_produces_both_values(self) -> None:
        """Label.sample() produces both -1 and 1 over many draws."""
        samples = {Label.sample() for _ in range(200)}
        assert -1 in samples
        assert 1 in samples


# ---------------------------------------------------------------------------
# Unit tests: pset factory functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreatePsetZigzag:
    """Unit tests for `create_pset_zigzag_*` factory functions."""

    def test_minimal_returns_typed_pset(self) -> None:
        """create_pset_zigzag_minimal() returns a PrimitiveSetTyped."""
        pset = create_pset_zigzag_minimal()
        assert isinstance(pset, gp.PrimitiveSetTyped)

    def test_medium_returns_typed_pset(self) -> None:
        """create_pset_zigzag_medium() returns a PrimitiveSetTyped."""
        pset = create_pset_zigzag_medium()
        assert isinstance(pset, gp.PrimitiveSetTyped)

    def test_large_returns_typed_pset(self) -> None:
        """create_pset_zigzag_large() returns a PrimitiveSetTyped."""
        pset = create_pset_zigzag_large()
        assert isinstance(pset, gp.PrimitiveSetTyped)

    def test_zigzag_primitive_registered_in_minimal(self) -> None:
        """zigzag_pivots is registered as a primitive in the minimal pset."""
        pset = create_pset_zigzag_minimal()
        all_names = {
            p.name
            for prims in pset.primitives.values()
            for p in prims
        }
        assert "zigzag_pivots" in all_names

    def test_zigzag_primitive_registered_in_medium(self) -> None:
        """zigzag_pivots is registered as a primitive in the medium pset."""
        pset = create_pset_zigzag_medium()
        all_names = {
            p.name
            for prims in pset.primitives.values()
            for p in prims
        }
        assert "zigzag_pivots" in all_names

    def test_medium_has_more_primitives_than_minimal(self) -> None:
        """Medium pset contains strictly more primitives than minimal pset."""
        minimal = create_pset_zigzag_minimal()
        medium = create_pset_zigzag_medium()
        count_minimal = sum(len(v) for v in minimal.primitives.values())
        count_medium = sum(len(v) for v in medium.primitives.values())
        assert count_medium > count_minimal

    def test_large_has_more_primitives_than_medium(self) -> None:
        """Large pset contains strictly more primitives than medium pset."""
        medium = create_pset_zigzag_medium()
        large = create_pset_zigzag_large()
        count_medium = sum(len(v) for v in medium.primitives.values())
        count_large = sum(len(v) for v in large.primitives.values())
        assert count_large > count_medium

    def test_pset_has_five_ohlcv_inputs(self) -> None:
        """Zigzag pset accepts exactly five OHLCV inputs."""
        pset = create_pset_zigzag_minimal()
        # pset.ins is the list of input types (Open, High, Low, Close, Volume)
        assert len(pset.ins) == 5


# ---------------------------------------------------------------------------
# Integration tests: zigzag labels with TreeOptimizer
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestZigzagWithOptimizer:
    """Integration tests for zigzag-derived labels used with `TreeOptimizer`."""

    @pytest.fixture(scope="class")
    def df_and_labels(self) -> tuple[pd.DataFrame, pd.Series]:
        """Shared OHLCV data and zigzag labels for optimizer integration tests."""
        df = generate_synthetic_ohlcv(500, 42)
        labels = zigzag_pivots(df["close"], 0.05, -1)
        assert isinstance(labels, pd.Series)
        return df, labels

    def test_fit_with_zigzag_labels_sets_population(
        self, df_and_labels: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """fit() with zigzag labels sets population_ to the expected size."""
        df, labels = df_and_labels
        mu = 6
        opt = TreeOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(F1Metric(),),
            mu=mu,
            lambda_=12,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(X=df, entry_label=labels)

        assert hasattr(opt, "population_")
        assert len(opt.population_) == mu

    def test_fit_with_zigzag_labels_produces_valid_logbook(
        self, df_and_labels: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """fit() with zigzag labels produces a logbook with generations+1 entries."""
        df, labels = df_and_labels
        generations = 2
        opt = TreeOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(F1Metric(),),
            mu=6,
            lambda_=12,
            generations=generations,
            seed=42,
            verbose=False,
        )
        opt.fit(X=df, entry_label=labels)

        assert len(opt.logbook_) == generations + 1

    def test_all_individuals_have_valid_fitness_after_fit(
        self, df_and_labels: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """All individuals in population_ have valid fitness values after fit()."""
        df, labels = df_and_labels
        opt = TreeOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(F1Metric(),),
            mu=6,
            lambda_=12,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(X=df, entry_label=labels)

        for ind in opt.population_:
            assert ind.fitness.valid

    def test_look_ahead_bias_label_length_consistency(
        self, df_and_labels: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Zigzag labels have the same length as the input series."""
        df, labels = df_and_labels
        assert len(labels) == len(df)

