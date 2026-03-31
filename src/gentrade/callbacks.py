"""Callback protocols and implementations for optimizer lifecycle hooks.

Callbacks allow users to plug custom behaviour into the GP evolution
process at key stages: fit start, generation end, and fit end.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gentrade.optimizer.base import BaseOptimizer


@runtime_checkable
class Callback(Protocol):
    """Lifecycle hooks for optimizer events.

    Implement this protocol to inject custom logic at specific points
    during evolution.
    """

    def on_fit_start(self, optimizer: "BaseOptimizer") -> None:
        """Called once at the start of fit(), before evolution begins."""
        ...

    def on_fit_end(self, optimizer: "BaseOptimizer") -> None:
        """Called once at the end of fit(), after evolution completes."""
        ...
