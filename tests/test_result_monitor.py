"""Unit tests for ResultMonitor behavior and message handling."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from deap import tools

from gentrade.algo_res import AlgorithmResult
from gentrade.island import (
    ErrorMessage,
    IslandCompletedMessage,
    ResultMessage,
    ResultMonitor,
)


def _make_result_msg(
    island_id: int = 0, generation: int = 1, **overrides: Any
) -> ResultMessage:
    """Create a default ResultMessage instance for ResultMonitor tests."""
    defaults: dict[str, Any] = {
        "island_id": island_id,
        "generation": generation,
        "best_individual": None,
        "best_fitness_val": None,
        "best_fit": (1.0,),
        "tree_height_mean": 3.5,
        "n_evaluated": 10,
        "eval_time": 0.1,
        "generation_time": 0.2,
        "population_size": 20,
    }
    defaults.update(overrides)
    return ResultMessage(**defaults)


def _make_error_msg(island_id: int = 0, **overrides: Any) -> ErrorMessage:
    """Create a default ErrorMessage instance for ResultMonitor tests."""
    defaults: dict[str, Any] = {
        "island_id": island_id,
        "error_type": "RuntimeError",
        "traceback": "Traceback ...",
    }
    defaults.update(overrides)
    return ErrorMessage(**defaults)


def _make_completed_msg(island_id: int = 0) -> IslandCompletedMessage[Any]:
    """Create a default IslandCompletedMessage for ResultMonitor tests."""
    return IslandCompletedMessage(
        island_id=island_id,
        result=AlgorithmResult(
            populations=[],
            logbooks=[],
            halloffames=[],
        ),
        # final_population=[],
        # final_logbook=tools.Logbook(),
        # final_hof=None,
    )


@pytest.mark.unit
class TestResultMonitorDispatch:
    """Verify ResultMonitor dispatches registered callbacks."""

    def test_per_message_callback_called(self) -> None:
        monitor = ResultMonitor[Any](n_islands=1)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        # pre-fill queue with two result messages and completion
        mq = monitor.master_queue
        mq.put(_make_result_msg(generation=0))
        mq.put(_make_result_msg(generation=1))
        mq.put(_make_completed_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc])
        # should be called for each ResultMessage
        assert mock_handler.on_island_generation_complete.call_count == 2

    def test_full_generation_callback(self) -> None:
        monitor = ResultMonitor[Any](n_islands=2)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        mq = monitor.master_queue
        # island 0 gen1, island1 gen1
        mq.put(_make_result_msg(island_id=0, generation=1))
        mq.put(_make_result_msg(island_id=1, generation=1))
        # completions
        mq.put(_make_completed_msg(island_id=0))
        mq.put(_make_completed_msg(island_id=1))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc, proc])
        # generation complete should be called once for generation 1
        calls = list(mock_handler.on_generation_complete.call_args_list)
        assert len(calls) == 1
        gen_arg = calls[0][0][0]
        assert gen_arg == 1

    def test_evolution_complete_callback(self) -> None:
        monitor = ResultMonitor[Any](n_islands=1)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        mq = monitor.master_queue
        mq.put(_make_result_msg(island_id=0, generation=0))
        mq.put(_make_completed_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc])
        # on_evolution_complete called once
        assert mock_handler.on_evolution_complete.call_count == 1


@pytest.mark.unit
class TestResultMonitorStateConsistency:
    """Tests to ensure core state logic is sound and doesn't hide bugs."""

    def test_results_stored_correctly(self) -> None:
        """Verify results are stored in _results_by_island for later retrieval."""
        monitor = ResultMonitor[Any](n_islands=1)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        mq = monitor.master_queue
        mq.put(_make_result_msg(island_id=0, generation=0))
        mq.put(_make_result_msg(island_id=0, generation=1))
        mq.put(_make_completed_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc])

        # Verify results are stored
        results = monitor.get_results()
        assert 0 in results
        assert len(results[0]) == 2
        assert results[0][0].generation == 0
        assert results[0][1].generation == 1

    def test_generation_complete_only_when_all_islands_report(self) -> None:
        """Verify on_generation_complete fires only after all islands report gen N."""
        monitor = ResultMonitor[Any](n_islands=2)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        mq = monitor.master_queue
        # Island 0 reports gen 1
        mq.put(_make_result_msg(island_id=0, generation=1))
        # Island 1 reports gen 1 - should trigger on_generation_complete
        mq.put(_make_result_msg(island_id=1, generation=1))
        # Both completions
        mq.put(_make_completed_msg(island_id=0))
        mq.put(_make_completed_msg(island_id=1))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc, proc])

        # Should be called exactly once (when both islands report gen 1)
        assert mock_handler.on_generation_complete.call_count == 1
        gen_arg = mock_handler.on_generation_complete.call_args_list[0][0][0]
        assert gen_arg == 1

    def test_generation_complete_only_when_all_report_same_gen(self) -> None:
        """Verify on_generation_complete fires when all islands report same gen."""
        monitor = ResultMonitor[Any](n_islands=2)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        mq = monitor.master_queue
        # Both islands report gen 0
        mq.put(_make_result_msg(island_id=0, generation=0))
        mq.put(_make_result_msg(island_id=1, generation=0))
        # Both completions
        mq.put(_make_completed_msg(island_id=0))
        mq.put(_make_completed_msg(island_id=1))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc, proc])

        # Should be called once for generation 0 (both islands have it)
        assert mock_handler.on_generation_complete.call_count == 1
        calls = list(mock_handler.on_generation_complete.call_args_list)
        assert calls[0][0][0] == 0

    def test_staggered_generations_fires_correctly(self) -> None:
        """Verify generation-complete fires only after all islands report.

        Island 0 races ahead to gen 2 while island 1 only reaches gen 1.
        Generation-complete should fire for gens 0 and 1 (both reported)
        but NOT for gen 2 (only island 0 reported it).
        """
        monitor = ResultMonitor[Any](n_islands=2)
        mock_handler = MagicMock()
        monitor.register_result_handler(mock_handler)

        mq = monitor.master_queue
        # Island 0 races ahead
        mq.put(_make_result_msg(island_id=0, generation=0))
        mq.put(_make_result_msg(island_id=0, generation=1))
        mq.put(_make_result_msg(island_id=0, generation=2))
        # Island 1 catches up partially
        mq.put(_make_result_msg(island_id=1, generation=0))
        mq.put(_make_result_msg(island_id=1, generation=1))
        # Both complete
        mq.put(_make_completed_msg(island_id=0))
        mq.put(_make_completed_msg(island_id=1))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc, proc])

        # gen 0 and gen 1 should fire; gen 2 should NOT
        calls = mock_handler.on_generation_complete.call_args_list
        fired_gens = {c[0][0] for c in calls}
        assert fired_gens == {0, 1}

    def test_multiple_generations_per_island_tracked(self) -> None:
        """Verify multiple generations from same island are all tracked."""
        monitor = ResultMonitor[Any](n_islands=1)
        mq = monitor.master_queue
        mq.put(_make_result_msg(island_id=0, generation=0, best_fit=(1.0,)))
        mq.put(_make_result_msg(island_id=0, generation=1, best_fit=(0.9,)))
        mq.put(_make_result_msg(island_id=0, generation=2, best_fit=(0.8,)))
        mq.put(_make_completed_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc])

        results = monitor.get_results()
        assert len(results[0]) == 3
        # Verify they're in order
        assert results[0][0].best_fit == (1.0,)
        assert results[0][1].best_fit == (0.9,)
        assert results[0][2].best_fit == (0.8,)

    def test_final_results_captured(self) -> None:
        """Verify final population and logbook are captured on completion."""
        monitor = ResultMonitor[Any](n_islands=1)
        mq = monitor.master_queue
        mq.put(_make_result_msg(island_id=0, generation=0))

        logbook = tools.Logbook()
        logbook.record(gen=0, nevals=10)
        final_pop = ["ind1", "ind2"]
        final_hof = tools.HallOfFame(maxsize=1)
        # Avoid the need for real individuals by using items list directly.
        final_hof.items.append(final_pop[0])
        mq.put(
            IslandCompletedMessage(
                island_id=0,
                result=AlgorithmResult.from_single_pop(  # type: ignore
                    population=final_pop,
                    logbook=logbook,
                    halloffame=final_hof,
                ),
            )
        )

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc])

        final_results = monitor.get_final_results()
        assert len(final_results) == 1
        # pop, log, hof = final_results[0]
        assert final_results[0].population == final_pop
        assert final_results[0].logbook == logbook
        hof = final_results[0].halloffame
        # Use equality, not identity (logbook copied through queue)
        assert hof is not None
        assert hof.items == final_hof.items == [final_pop[0]]

    def test_multiple_islands_completion_captured(self) -> None:
        """Verify all islands' completion data is captured."""
        monitor = ResultMonitor[Any](n_islands=2)
        mq = monitor.master_queue

        for island_id in [0, 1]:
            mq.put(_make_result_msg(island_id=island_id, generation=0))
            final_pop = [f"ind_{island_id}_1", f"ind_{island_id}_2"]
            logbook = tools.Logbook()
            logbook.header = ["island", "gen", "nevals"]
            logbook.record(island=island_id, gen=0, nevals=10)
            final_hof = tools.HallOfFame(maxsize=1)
            # Avoid the need for real individuals by using items list directly.
            final_hof.items.append(final_pop[island_id])

            mq.put(
                IslandCompletedMessage(
                    island_id=island_id,
                    result=AlgorithmResult.from_single_pop(  # type: ignore
                        population=final_pop,
                        logbook=logbook,
                        halloffame=final_hof,
                    ),
                )
            )

        proc = MagicMock()
        proc.is_alive.return_value = False

        monitor.wait(processes=[proc, proc])

        final_results = monitor.get_final_results()
        assert len(final_results) == 2
        # Check population
        assert final_results[0].population == ["ind_0_1", "ind_0_2"]
        assert final_results[1].population == ["ind_1_1", "ind_1_2"]
        # Check logbook
        assert final_results[0].logbook[0]["island"] == 0
        assert final_results[1].logbook[0]["island"] == 1
        # Check hof
        assert final_results[0].halloffame is not None
        assert final_results[1].halloffame is not None
        assert final_results[0].halloffame[0] == "ind_0_1"
        assert final_results[1].halloffame[0] == "ind_1_2"


@pytest.mark.unit
class TestResultMonitorErrors:
    """Verify ResultMonitor errors are surfaced and handled correctly."""

    def test_error_triggers_runtime_error(self) -> None:
        monitor = ResultMonitor[Any](n_islands=1)
        mq = monitor.master_queue
        mq.put(_make_error_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        with pytest.raises(RuntimeError):
            monitor.wait(processes=[proc])

    def test_dead_processes_without_completion_raises(self) -> None:
        """Verify monitor raises when all processes die without sending completions."""
        monitor = ResultMonitor[Any](n_islands=2)

        proc = MagicMock()
        proc.is_alive.return_value = False

        with pytest.raises(RuntimeError, match="never sent completion"):
            monitor.wait(processes=[proc, proc], timeout=0.01)

    def test_dead_processes_partial_completion_raises(self) -> None:
        """Verify monitor raises when processes die with only some islands completed."""
        monitor = ResultMonitor[Any](n_islands=2)
        mq = monitor.master_queue
        # Only island 0 completes; island 1 never sends anything
        mq.put(_make_result_msg(island_id=0, generation=0))
        mq.put(_make_completed_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        with pytest.raises(RuntimeError, match="never sent completion"):
            monitor.wait(processes=[proc, proc], timeout=0.01)

    def test_duplicate_completion_raises(self) -> None:
        monitor = ResultMonitor[Any](n_islands=1)
        mq = monitor.master_queue
        mq.put(_make_completed_msg(island_id=0))
        mq.put(_make_completed_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        with pytest.raises(RuntimeError):
            monitor.wait(processes=[proc])

    def test_result_after_completion_raises(self) -> None:
        monitor = ResultMonitor[Any](n_islands=1)
        mq = monitor.master_queue
        mq.put(_make_completed_msg(island_id=0))
        mq.put(_make_result_msg(island_id=0, generation=1))

        proc = MagicMock()
        proc.is_alive.return_value = False

        with pytest.raises(RuntimeError):
            monitor.wait(processes=[proc])

    def test_error_handler_exception_still_raises_original(self) -> None:
        monitor = ResultMonitor[Any](n_islands=1)

        class BadErr:
            def on_error(self, err: ErrorMessage) -> None:
                raise RuntimeError("handler boom")

        monitor.register_error_handler(BadErr())
        mq = monitor.master_queue
        mq.put(_make_error_msg(island_id=0))

        proc = MagicMock()
        proc.is_alive.return_value = False

        with pytest.raises(RuntimeError):
            monitor.wait(processes=[proc])
