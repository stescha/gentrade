"""Migration payload contracts for island-model algorithms.

This module defines typed payload packets used when transferring individuals
between islands. The packet format is algorithm-defined: the island runtime
treats payloads as opaque objects and delegates integration to the algorithm's
migration hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from gentrade.types import IndividualT, PairTreeComponent

# Unclear if the base class should be kept. Maybe replace with a protocol or just use
# the concrete class directly maybe with generics if needed. For now we keep it as
# a marker and for potential future shared fields.


@dataclass(frozen=True)
class MigrationPacket:
    """Migration payload exchanged between islands.

    The island runtime pushes and pulls instances of this dataclass through
    :class:`~gentrade.island.QueueDepot` without inspecting the ``data``
    field.
    """


@dataclass(frozen=True)
class MultiPopMigrationPacket(MigrationPacket):
    """
    Args:
        data: Dict mapping species index to the corresponding component
          (gp.PrimitiveTree) from the emigrant individual. The island runtime treats
          this as an opaque payload and does not inspect or interpret the contents.
          The structure is defined by the algorithm and is designed to support
          multi-population representations where each species corresponds to a
          component of the overall individual (list).
    """

    # Sup population (species) index to list of components
    data: dict[int, PairTreeComponent]


@dataclass(frozen=True)
class SinglePopMigrationPacket(MigrationPacket, Generic[IndividualT]):
    data: IndividualT
