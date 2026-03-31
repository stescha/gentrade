"""Migration payload contracts for island-model algorithms.

This module defines typed payload packets used when transferring individuals
between islands. The packet format is algorithm-defined: the island runtime
treats payloads as opaque objects and delegates integration to the algorithm's
migration hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from gentrade.types import IndividualT


@dataclass(frozen=True)
class MigrationPacket(Generic[IndividualT]):
    """Typed migration payload exchanged between islands.

    The island runtime pushes and pulls instances of this dataclass through
    :class:`~gentrade.island.QueueDepot` without inspecting the ``data``
    field. Consumers (algorithms) are responsible for validating
    ``payload_type`` and the keys present in ``data``.

    Args:
        payload_type: String tag identifying the algorithm-specific payload
            format. Must be validated by the consuming algorithm.
        data: Dict mapping string keys to individual lists. For ACC use
            keys ``"entry"`` and ``"exit"``.
    """

    payload_type: str
    data: dict[str, list[IndividualT]]
