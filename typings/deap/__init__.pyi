"""Type stubs for DEAP library.

DEAP (Distributed Evolutionary Algorithms in Python) is an evolutionary
computation framework. Since DEAP is not typed, these stubs provide
minimal but functional type hints for commonly used components.
"""

from deap import algorithms, base, creator, gp, tools

__all__ = ["base", "creator", "gp", "tools", "algorithms"]
