"""Individual class hierarchy for gentrade GP optimizers.

This module provides a lightweight, stable individual container for GP
trees. Each individual uses a DEAP-created Fitness class with the correct
number of objectives.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable, Tuple, cast

from deap import base, creator, gp

# Cache for created Fitness classes, keyed by exact weights tuple.
_fitness_classes_cache: dict[tuple[float, ...], type[base.Fitness]] = {}


def _get_or_create_fitness_class(weights: Tuple[float, ...]) -> type[base.Fitness]:
    """Retrieve or create a Fitness class for the given objective weights.

    Caches classes by the exact weights tuple to preserve maximization/minimization
    intent (signs of weights). The created classes are registered in DEAP's
    :mod:`deap.creator` to ensure pickling works across processes.

    Args:
        weights: Tuple of objective weights (signs indicate minimization).

    Returns:
        A :class:`deap.base.Fitness` subclass configured with the provided
        weights.
    """
    if weights in _fitness_classes_cache:
        return _fitness_classes_cache[weights]

    # Create a unique class name based on weights
    # Replace '-' with 'm' and '.' with 'p' to make a valid identifier
    class_name = "Fitness_" + "_".join(
        str(w).replace("-", "m").replace(".", "p") for w in weights
    )

    # Check if already present in deap.creator namespace
    if hasattr(creator, class_name):
        cls = getattr(creator, class_name)
        _fitness_classes_cache[weights] = cls
        return cast(type[base.Fitness], cls)

    # Create via DEAP's creator using the requested weights
    creator.create(class_name, base.Fitness, weights=weights)
    cls = getattr(creator, class_name)
    _fitness_classes_cache[weights] = cls
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
        fitness_cls = _get_or_create_fitness_class(weights)
        self.fitness = fitness_cls()


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


class PairTreeIndividual(TreeIndividualBase):
    """A GP individual containing exactly two primitive trees: buy and sell.

    The first tree (index 0) generates entry signals; the second (index 1)
    generates exit signals. Both trees share the same primitive set.

    Attributes:
        buy_tree: The entry-signal tree (``self[0]``).
        sell_tree: The exit-signal tree (``self[1]``).
    """

    def __init__(
        self,
        content: Iterable[gp.PrimitiveTree],
        weights: Tuple[float, ...],
    ) -> None:
        """Initialize a pair-tree individual.

        Args:
            content: Iterable of exactly two :class:`deap.gp.PrimitiveTree`
                instances: buy tree first, sell tree second.
            weights: Fitness objective weights (length determines objective count).

        Raises:
            ValueError: If ``content`` does not contain exactly two trees.
        """
        trees = list(content)
        if len(trees) != 2:
            raise ValueError(
                f"PairTreeIndividual requires exactly 2 trees, got {len(trees)}."
            )
        super().__init__(trees, weights)

    @property
    def buy_tree(self) -> gp.PrimitiveTree:
        """Return the buy (entry) tree at index 0."""
        return self[0]

    @property
    def sell_tree(self) -> gp.PrimitiveTree:
        """Return the sell (exit) tree at index 1."""
        return self[1]


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
