# THis file needs rework!!

# """Unit tests for island lifecycle handler implementations."""

# from __future__ import annotations

# import copy
# import multiprocessing as mp
# import time
# from queue import Queue as ThreadQueue
# from typing import Any, cast
# from unittest.mock import MagicMock

# import pytest
# from deap import base, gp, tools

# from gentrade.acc import AccEa
# from gentrade.algorithms import AlgorithmState, StopEvolution
# from gentrade.individual import PairTreeIndividual, TreeIndividual, TreeIndividualBase
# from gentrade.island import (
#     ACC_COMPONENTS_PAYLOAD,
#     ErrorMessage,
#     IslandCompletedMessage,
#     QueueDepot,
#     ResultMessage,
#     _IslandDescriptor,
#     _StandardIslandLifecycleHandler,
# )
# from gentrade.migration import MigrationPacket
# from gentrade.minimal_pset import create_pset_zigzag_minimal
# from gentrade.topologies import MigrationTopology


# class _StaticTopology(MigrationTopology):
#     def __init__(self, plan: list[tuple[int, int]]):
#         self._plan = plan

#     def get_immigrants(self, island_id: int) -> list[tuple[int, int]]:
#         return self._plan


# _STD_WEIGHTS = (1.0,)


# class _HandlerIndividual(TreeIndividualBase):
#     def __init__(self, value: float) -> None:
#         super().__init__([], _STD_WEIGHTS)
#         self.fitness.values = (value,)


# def _make_std_individual(value: float) -> TreeIndividualBase:
#     return _HandlerIndividual(value)


# @pytest.fixture
# def std_toolbox() -> base.Toolbox:
#     tb = base.Toolbox()
#     tb.register("clone", copy.deepcopy)

#     def _sel_best(pop: list[TreeIndividualBase], k: int) -> list[TreeIndividualBase]:
#         return sorted(
#             pop,
#             key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -1e9,
#             reverse=True,
#         )[:k]

#     def _sel_worst(pop: list[TreeIndividualBase], k: int) -> list[TreeIndividualBase]:
#         return sorted(
#             pop,
#             key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -1e9,
#         )[:k]

#     tb.register("select_emigrants", _sel_best)
#     tb.register("select_best", _sel_best)
#     tb.register("select_replace", _sel_worst)
#     return tb


# @pytest.fixture
# def std_handler_env(std_toolbox: base.Toolbox) -> dict[str, Any]:
#     depots = [QueueDepot(maxlen=10), QueueDepot(maxlen=10)]
#     descriptor = _IslandDescriptor(
#         island_id=0,
#         depot=depots[0],
#         neighbor_depots=depots,
#     )
#     topology = _StaticTopology(plan=[(1, 2)])
#     stop_event = mp.Event()
#     thread_queue: ThreadQueue[ResultMessage | ErrorMessage | IslandCompletedMessage] = (
#         ThreadQueue()
#     )
#     master_queue = cast(Any, thread_queue)

#     def _evaluate(
#         toolbox: base.Toolbox,
#         individuals: list[TreeIndividualBase],
#         all_: bool = False,
#     ) -> tuple[int, float]:
#         for idx, ind in enumerate(individuals):
#             ind.fitness.values = (99.0 + idx,)
#         return len(individuals), 0.01

#     algorithm = MagicMock()
#     algorithm.evaluate_individuals.side_effect = _evaluate

#     handler = _StandardIslandLifecycleHandler(
#         algorithm=algorithm,
#         descriptor=descriptor,
#         topology=topology,
#         stop_event=stop_event,
#         master_queue=master_queue,
#         migration_rate=1,
#         migration_count=2,
#         pull_timeout=0.01,
#         pull_max_retries=1,
#         push_timeout=0.001,
#     )
#     return {
#         "handler": handler,
#         "descriptor": descriptor,
#         "depots": depots,
#         "topology": topology,
#         "stop_event": stop_event,
#         "master_queue": thread_queue,
#         "toolbox": std_toolbox,
#     }


# @pytest.mark.unit
# def test_standard_handler_integrates_and_reports(
#     std_handler_env: dict[str, Any],
# ) -> None:
#     handler: _StandardIslandLifecycleHandler = std_handler_env["handler"]
#     descriptor: _IslandDescriptor = std_handler_env["descriptor"]
#     depots: list[QueueDepot] = std_handler_env["depots"]
#     master_queue: ThreadQueue[ResultMessage | ErrorMessage | IslandCompletedMessage] = (
#         std_handler_env["master_queue"]
#     )
#     toolbox: base.Toolbox = std_handler_env["toolbox"]

#     population: list[TreeIndividualBase] = [
#         _make_std_individual(v) for v in (0.1, 0.2, 0.3)
#     ]
#     immigrants: list[TreeIndividualBase] = [_make_std_individual(v) for v in (1.0, 2.0)]
#     depots[1].push(immigrants)

#     init_state = AlgorithmState(
#         generation=0,
#         logbook=tools.Logbook(),
#         halloffame=None,
#         n_evaluated=len(population),
#         eval_time=0.0,
#     )
#     handler.post_initialization(population, init_state, toolbox)
#     init_msg = master_queue.get_nowait()
#     assert isinstance(init_msg, ResultMessage)
#     assert init_msg.generation == 0
#     assert init_msg.n_emigrants == 2
#     descriptor.depot.pull(2)

#     state = AlgorithmState(
#         generation=1,
#         logbook=tools.Logbook(),
#         halloffame=None,
#     )
#     population = handler.pre_generation(population, state, toolbox)
#     assert state.n_immigrants == 2
#     assert all(ind.fitness.valid for ind in population)

#     state.loop_start_time = time.perf_counter()
#     state.best_individual = population[0]
#     state.best_fit = population[0].fitness.values
#     state.n_evaluated = len(population)
#     state.eval_time = 0.5

#     population = handler.post_generation(population, state, toolbox)
#     assert state.n_emigrants == 2
#     result_msg = master_queue.get_nowait()
#     assert isinstance(result_msg, ResultMessage)
#     assert result_msg.n_immigrants == 2
#     assert result_msg.n_emigrants == 2
#     pulled = descriptor.depot.pull(2)
#     assert len(pulled) == 2


# def test_standard_handler_respects_stop_event(std_handler_env: dict[str, Any]) -> None:
#     handler: _StandardIslandLifecycleHandler = std_handler_env["handler"]
#     std_handler_env["stop_event"].set()
#     state = AlgorithmState(
#         generation=1,
#         logbook=tools.Logbook(),
#         halloffame=None,
#     )
#     with pytest.raises(StopEvolution):
#         handler.pre_generation([], state, std_handler_env["toolbox"])


# @pytest.fixture
# def acc_toolbox() -> base.Toolbox:
#     from gentrade.optimizer.tree import _create_tree_toolbox

#     toolbox = _create_tree_toolbox(
#         pset=create_pset_zigzag_minimal(),
#         mutation=gp.mutUniform,  # type: ignore[arg-type]
#         mutation_params=None,
#         crossover=gp.cxOnePoint,
#         crossover_params=None,
#         selection=tools.selTournament,  # type: ignore[arg-type]
#         selection_params={"tournsize": 2},
#         select_best=tools.selBest,  # type: ignore[arg-type]
#         select_best_params=None,
#         select_replace=tools.selWorst,  # type: ignore[arg-type]
#         select_replace_params=None,
#         select_emigrants=tools.selBest,  # type: ignore[arg-type]
#         select_emigrants_params=None,
#         tree_min_depth=1,
#         tree_max_depth=3,
#         tree_max_height=7,
#         tree_gen="grow",
#     )
#     toolbox.register("map", map)
#     toolbox.register("evaluate", lambda ind: (0.1,))
#     return toolbox


# @pytest.fixture
# def acc_ea() -> AccEa:
#     mock_evaluator = MagicMock()
#     mock_metric = MagicMock()
#     mock_metric.weight = 1.0
#     mock_evaluator.metrics = (mock_metric,)
#     return AccEa(
#         mu=2,
#         lambda_=2,
#         cxpb=0.5,
#         mutpb=0.2,
#         ngen=2,
#         evaluator=mock_evaluator,
#     )


# def _make_acc_components(acc_ea: AccEa, toolbox: base.Toolbox) -> None:
#     from gentrade.growtree import genGrow

#     pset = create_pset_zigzag_minimal()
#     entry = []
#     exit_ = []
#     for _ in range(acc_ea.mu):
#         entry_nodes: Any = genGrow(pset, min_=1, max_=2)  # type: ignore[no-untyped-call]
#         exit_nodes: Any = genGrow(pset, min_=1, max_=2)  # type: ignore[no-untyped-call]
#         e = TreeIndividual([gp.PrimitiveTree(entry_nodes)], (1.0,))
#         x = TreeIndividual([gp.PrimitiveTree(exit_nodes)], (1.0,))
#         e.fitness.values = (0.1,)
#         x.fitness.values = (0.2,)
#         entry.append(e)
#         exit_.append(x)
#     acc_ea.entry_population = entry
#     acc_ea.exit_population = exit_


# def test_acc_handler_rejects_invalid_packets(
#     acc_ea: AccEa,
#     acc_toolbox: base.Toolbox,
# ) -> None:
#     depots = [QueueDepot(maxlen=5)]
#     descriptor = _IslandDescriptor(
#         island_id=0,
#         depot=depots[0],
#         neighbor_depots=depots,
#     )
#     handler = _StandardIslandLifecycleHandler(
#         algorithm=acc_ea,
#         descriptor=descriptor,
#         topology=_StaticTopology(plan=[]),
#         stop_event=mp.Event(),
#         master_queue=MagicMock(),
#         migration_rate=1,
#         migration_count=1,
#         pull_timeout=0.01,
#         pull_max_retries=1,
#         push_timeout=0.001,
#     )
#     population: list[PairTreeIndividual] = []
#     bad_packet: MigrationPacket[Any] = MigrationPacket(
#         payload_type="wrong",
#         data={"entry": [], "exit": []},
#     )
#     with pytest.raises(ValueError, match="Unknown payload_type"):
#         handler._accept_immigrants(population, [bad_packet], acc_toolbox)


# @pytest.mark.skip(reason="_AccIslandLifecycleHandler removed, unified migration now uses algorithm hooks")
# def test_acc_handler_integrates_packets_preserving_shape(
#     acc_ea: AccEa,
#     acc_toolbox: base.Toolbox,
# ) -> None:
#     depots = [QueueDepot(maxlen=5), QueueDepot(maxlen=5)]
#     descriptor = _IslandDescriptor(
#         island_id=0,
#         depot=depots[0],
#         neighbor_depots=depots,
#     )
#     handler = _StandardIslandLifecycleHandler(
#         algorithm=acc_ea,
#         descriptor=descriptor,
#         topology=_StaticTopology(plan=[]),
#         stop_event=mp.Event(),
#         master_queue=MagicMock(),
#         migration_rate=1,
#         migration_count=1,
#         pull_timeout=0.01,
#         pull_max_retries=1,
#         push_timeout=0.001,
#     )
#     _make_acc_components(acc_ea, acc_toolbox)
#     population = acc_ea.assemble_current_population()

#     packet = MigrationPacket(
#         payload_type=ACC_COMPONENTS_PAYLOAD,
#         data={
#             "entry": [acc_toolbox.clone(acc_ea.entry_population[0])],
#             "exit": [acc_toolbox.clone(acc_ea.exit_population[0])],
#         },
#     )

#     new_population = handler._accept_immigrants(population, [packet], acc_toolbox)
#     assert len(new_population) == acc_ea.mu
#     assert all(isinstance(ind, PairTreeIndividual) for ind in new_population)
