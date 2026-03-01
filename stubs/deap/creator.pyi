"""Type stubs for deap.creator module.

The creator module works by creating classes dynamically using the ``create``
function. After calling ``creator.create("ClassName", base, **attrs)``, you
can access the created class as ``creator.ClassName``.

This stub module enables that pattern by allowing dynamic attribute access.
"""

from typing import Any

# Dynamically created classes (stubs for commonly-created classes)
# After calling creator.create(), the classes are accessible as module attributes
FitnessMax: type
FitnessMin: type
FitnessMulti: type
Individual: type

def create(
    name: str,
    base: type | tuple[type, ...],
    **attributes: Any,
) -> type:
    """Create a new class with the given name and attributes.

    The created class becomes accessible as an attribute on the creator module.

    Args:
        name: Name of the class to create.
        base: Base class or tuple of base classes.
        **attributes: Class attributes and constraints to set.

    Returns:
        The newly created class.

    Example:
        >>> creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        >>> cls = creator.FitnessMax  # Now accessible
    """
    ...
