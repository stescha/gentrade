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


class TreeIndividualBase(list[gp.PrimitiveTree]):
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


class PairIndividual(TreeIndividualBase):
    """A GP individual containing exactly two primitive trees with fitness tracking.

    Represents a pair strategy where one tree generates entry (buy) signals and
    the other generates exit (sell) signals. Both trees operate on the same OHLCV
    data and produce boolean Series.

    Attributes:
        fitness: A :class:`deap.base.Fitness` instance with weights matching
            the number of optimization objectives.
    """

    def __init__(
        self,
        content: Iterable[gp.PrimitiveTree],
        weights: Tuple[float, ...],
    ) -> None:
        """Initialize a pair individual with exactly two trees and fitness weights.

        Args:
            content: Iterable of exactly two :class:`deap.gp.PrimitiveTree`
                instances; ``content[0]`` is the buy tree and ``content[1]``
                is the sell tree.
            weights: Tuple of objective weights used to create the Fitness instance.

        Raises:
            ValueError: If content does not contain exactly two trees.
        """
        trees = list(content)
        if len(trees) != 2:
            raise ValueError(
                f"PairIndividual must contain exactly two trees, got {len(trees)}"
            )
        super().__init__(trees, weights)

    @property
    def buy_tree(self) -> gp.PrimitiveTree:
        """Return the buy (entry signal) tree.

        Returns:
            The first :class:`deap.gp.PrimitiveTree` (index 0).
        """
        return self[0]

    @property
    def sell_tree(self) -> gp.PrimitiveTree:
        """Return the sell (exit signal) tree.

        Returns:
            The second :class:`deap.gp.PrimitiveTree` (index 1).
        """
        return self[1]


class TreeIndividual(TreeIndividualBase):
    """A GP individual containing exactly one primitive tree with fitness tracking."""

    def __init__(
        self,
        content: Iterable[gp.PrimitiveTree] | gp.PrimitiveTree,
        weights: Tuple[float, ...],
    ) -> None:
        """Initialize a single-tree individual with given tree and fitness weights."""
        if isinstance(content, gp.PrimitiveTree):
            content = [content]
        elif not isinstance(content, list):
            raise ValueError(
                "TreeIndividual content must be a PrimitiveTree or list of "
                f"PrimitiveTrees, got {type(content)}"
            )
        elif len(content) != 1:
            raise ValueError(
                f"TreeIndividual must contain exactly one tree, got {len(content)}"
            )
        super().__init__(content, weights)

    @property
    def tree(self) -> gp.PrimitiveTree:
        """Return the tree for single-tree individuals.

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
