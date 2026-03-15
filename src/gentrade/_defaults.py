"""Shared default names used across the gentrade package."""

#: canonical key for the primary OHLCV dataset
from deap import tools

KEY_OHLCV = "ohlcv"

# -- Selection Operator Compatibility --
# Hardcoded lists of DEAP selection functions and their objective compatibility.
# In multi-objective optimization, individuals have a tuple of fitnesses.
# NSGA2 and SPEA2 are valid choices.

SELECTION_SINGLE_OBJ = {
    tools.selTournament,
    tools.selDoubleTournament,
    tools.selBest,
    tools.selWorst,
    tools.selRandom,
    tools.selRoulette,
}

SELECTION_MULTI_OBJ = {
    tools.selNSGA2,
    tools.selSPEA2,
}
