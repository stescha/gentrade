"""Individual class hierarchy for gentrade GP optimizers.

This module provides a lightweight, stable individual container for GP
trees. Each individual uses a DEAP-created Fitness class with the correct
number of objectives.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable, Tuple, cast

from deap import base, creator, gp

# Cache for created Fitness classes, keyed by objective count.
_fitness_classes_cache: dict[int, type[base.Fitness]] = {}


def _get_or_create_fitness_class(n_objectives: int) -> type[base.Fitness]:
    """Retrieve or create a Fitness class for the given number of objectives.

    This function manages a cache of Fitness classes keyed by objective count.
    It uses DEAP's :class:`deap.creator` to dynamically create Fitness classes
    with the correct number of objectives, ensuring they can be pickled and
    unpickled by worker processes in multiprocessing scenarios.

    The created classes follow the naming convention `FitnessMulti_{n_objectives}`
    and are registered in the `deap.creator` namespace for proper serialization.

    Args:
        n_objectives: Number of objectives (determines fitness tuple length).

    Returns:
        A :class:`deap.base.Fitness` class configured for the given number
        of objectives, with all weights set to 1.0 (maximization).
    """
    if n_objectives in _fitness_classes_cache:
        return _fitness_classes_cache[n_objectives]

    # Create a unique class name
    class_name = f"FitnessMulti_{n_objectives}"

    # Check if already created in deap.creator namespace
    if hasattr(creator, class_name):
        cls = getattr(creator, class_name)
        _fitness_classes_cache[n_objectives] = cls
        return cast(type[base.Fitness], cls)
    # Create via DEAP's creator with placeholder weights
    weights = tuple(1.0 for _ in range(n_objectives))
    creator.create(class_name, base.Fitness, weights=weights)
    cls = getattr(creator, class_name)
    _fitness_classes_cache[n_objectives] = cls
    return cast(type[base.Fitness], cls)


class TreeIndividual(list[gp.PrimitiveTree]):
    """A GP individual that wraps one or more primitive trees with fitness tracking.

    This class extends a list to contain :class:`deap.gp.PrimitiveTree` instances
    and attaches a :class:`deap.base.Fitness` object. It is the primary container
    for individuals throughout the optimization pipeline, replacing bare
    :class:`deap.gp.PrimitiveTree` objects to improve code clarity and enable
    seamless fitness management across single- and multi-objective scenarios.

    Attributes:
        fitness: A :class:`deap.base.Fitness` instance with weights matching
            the number of optimization objectives. Automatically created during
            initialization and managed per-individual to support fitness
            tuple resizing without global DEAP creator complications.
    """

    fitness: base.Fitness

    def __init__(
        self, content: Iterable[gp.PrimitiveTree], weights: Tuple[float, ...]
    ) -> None:
        """Initialize a tree individual with given trees and fitness weights.

        Args:
            content: Iterable of :class:`deap.gp.PrimitiveTree` instances
                to be stored in this individual. For single-tree optimization,
                this is typically a list with one tree.
            weights: Tuple of objective weights used to create the Fitness
                instance. Positive weights indicate objectives to maximize,
                negative weights indicate objectives to minimize. The length
                determines the number of objectives.
        """
        super().__init__(content)
        fitness_cls = _get_or_create_fitness_class(len(weights))
        self.fitness = fitness_cls()

    @property
    def tree(self) -> gp.PrimitiveTree:
        """Return the first (primary) tree for single-tree individuals.

        This property provides convenient access to the most common case:
        a single-tree GP individual. It returns `self[0]`.

        Returns:
            The first :class:`deap.gp.PrimitiveTree` in this individual.

        Raises:
            IndexError: If the individual contains no trees.
        """
        return self[0]


def apply_operators(
    tree_op: Callable[..., Any],
) -> Callable[..., tuple[TreeIndividual, ...]]:
    """Wrap a tree-level DEAP operator to work on `TreeIndividual` instances.

    This decorator lifts tree-level operators (e.g., `gp.cxOnePoint`, `gp.mutUniform`)
    to the individual level. It applies the wrapped operator to each corresponding
    tree position across all provided individuals, handling both single-tree and
    multi-tree individuals transparently.

    The wrapper operates on the minimum tree count across inputs to avoid
    `IndexError` when individuals have differing tree counts (e.g., after certain
    mutation operators).

    Args:
        tree_op: A callable that operates on GP trees
            (e.g., crossover or mutation functions from DEAP).

    Returns:
        A wrapped function that takes `TreeIndividual` instances, applies
        `tree_op` to their trees, and returns the modified individuals.
    """

    @wraps(tree_op)
    def wrapper(*individuals: TreeIndividual) -> tuple[TreeIndividual, ...]:
        if not individuals:
            return individuals
        # Operate only over the tree positions present in all individuals.
        # Some code paths may produce individuals with differing tree counts
        # (e.g., empty trees); use the minimum length to avoid IndexError.
        n_trees = min(len(ind) for ind in individuals)
        for i in range(n_trees):
            raw = tree_op(*[ind[i] for ind in individuals])
            results: list[gp.PrimitiveTree] | tuple[gp.PrimitiveTree, ...]
            results = raw if isinstance(raw, (list, tuple)) else (raw,)
            for ind, new_tree in zip(individuals, results, strict=True):
                ind[i] = new_tree
        return individuals

    return wrapper
