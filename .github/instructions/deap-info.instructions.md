# DEAP Library Reference: Genetic Programming & Evolutionary Algorithms

## 1. Core Architecture & Philosophy
DEAP relies on **functional composition** and **dynamic type creation**, not rigid class hierarchies.
*   **Decoupling:** Data structures (`Individual`), Operators (`Toolbox`), and Algorithms are separate.
*   **Explicit Registration:** You must strictly register every component in the `Toolbox`.

### 1.1 Dynamic Type Creation (`deap.creator`)
Generates classes at runtime.
*   **`creator.create(name, base, **kwargs)`**:
    *   **Fitness:** `weights` tuple defines objectives.
        *   `(1.0,)`: Maximize Single Objective.
        *   `(-1.0, 1.0)`: Multi-Objective (Min, Max).
    *   **Individual (GP):** Must inherit from `gp.PrimitiveTree`.
    *   **Individual (GA):** Inherits from `list`, `array.array`, or `set`.

### 1.2 The Toolbox (`deap.base.Toolbox`)
A dependency injection container.
*   **`toolbox.register(alias, method, *args, **kwargs)`**: Registers `method` as `alias` with fixed defaults.
*   **`toolbox.decorate(alias, decorator)`**: Wraps an operator (e.g., for bounds/bloat control).

## 2. Genetic Programming Components (`deap.gp`)

### 2.1 Primitive Sets (Vocabulary)
Defines the building blocks of the program tree.
*   **`gp.PrimitiveSetTyped(name, in_types, ret_type)`** (Mandatory for Logic/Trading):
    *   Enforces type safety (e.g., `bool` vs `float`).
*   **`pset.addPrimitive(func, [in_types], ret_type)`**: Registers a function.
*   **`pset.addTerminal(value, type)`**: Registers a static constant or input variable.
*   **`pset.addEphemeralConstant(name, generator, type)`**: Registers a random constant generator (e.g., `lambda: random.uniform(-1, 1)`).
*   **`pset.renameArguments(ARG0='x', ...)`**: Maps internal names to readable strings.

### 2.2 Tree Generation
*   **`gp.genFull(pset, min_, max_)`**: Max depth reached on all branches.
*   **`gp.genGrow(pset, min_, max_)`**: Variable depth (branches can end early).
*   **`gp.genHalfAndHalf(pset, min_, max_)`**: **Best Practice.** 50% Full, 50% Grow.

### 2.3 Compilation
*   **`gp.compile(expr, pset)`**: Compiles the tree expression into a callable Python function.
    *   *Performance Note:* Uses `eval`. For large datasets, ensure Primitives support `numpy` vectorization to avoid row-by-row loops.

## 3. Comprehensive Operator Reference (`deap.tools`, `deap.gp`)

### 3.1 Selection Operators
*   **Standard:**
    *   **`selTournament(ind, k, tournsize)`**: **Robust Default.** Selects best of `tournsize` random individuals.
    *   **`selRoulette(ind, k)`**: Probability proportional to fitness (requires positive fitness).
    *   **`selBest(ind, k)`** / **`selWorst(ind, k)`**: Deterministic top/bottom selection.
    *   **`selRandom(ind, k)`**: Random selection.
*   **Multi-Objective:**
    *   **`selNSGA2(ind, k)`**: **Standard.** Sorts by non-dominated sorting & crowding distance.
    *   **`selNSGA3(ind, k)`**: Optimized for Many-Objective (3+) problems using reference points.
    *   **`selSPEA2(ind, k)`**: Strength Pareto Evolutionary Algorithm 2.
*   **GP Specific:**
    *   **`selDoubleTournament(ind, k, fitness_size, parsimony_size, ...)`**: Selection for fitness, then for size. **Critical for Bloat Control.**
    *   **`selLexicase(ind, k)`**: Selects individuals that solve specific test cases unique from others. Excellent for symbolic regression.

### 3.2 Crossover (Mating) Operators
*   **Genetic Programming (Tree):**
    *   **`gp.cxOnePoint(ind1, ind2)`**: Swaps subtrees at random nodes.
    *   **`gp.cxOnePointLeafBiased(ind1, ind2, termpb)`**: Biased towards leaves (10% default) to reduce destructive root changes.
    *   **`gp.cxSemantic(ind1, ind2)`**: Geometric semantic crossover.
*   **Standard GA (Sequence):**
    *   **`cxTwoPoint`**, **`cxUniform`**: Standard sequence swaps.
    *   **`cxSimulatedBinary` (SBX)**, **`cxBlend`**: For floating point arrays.

### 3.3 Mutation Operators
*   **Genetic Programming (Tree):**
    *   **`gp.mutUniform(ind, expr, pset)`**: Replaces subtree with random new one.
    *   **`gp.mutShrink(ind)`**: Replaces branch with one of its arguments (Reduces size).
    *   **`gp.mutNodeReplacement(ind, pset)`**: Swaps function node with same arity.
    *   **`gp.mutInsert(ind, pset)`**: Inserts new branch at random position.
    *   **`gp.mutEphemeral(ind, mode)`**: Perturbs values of constant numbers inside the tree.
*   **Standard GA (Sequence):**
    *   **`mutGaussian`**, **`mutPolynomialBounded`**: For floating point arrays.
    *   **`mutFlipBit`**: For binary sequences.

### 3.4 Bloat Control & Constraints
*   **`gp.staticLimit(key, max_value)`**:
    *   Decorator for `mate` or `mutate`.
    *   **Mandatory:** If child > `max_value` (height/depth), the operation is reverted.
*   **`DeltaPenalty`**, **`ClosestValidPenalty`**: Fitness penalties for constraint violations.

### 3.5 Migration (Island Model)
*   **`migRing(populations, k, selection, ...)`**: Moves `k` individuals between sub-populations in a ring topology.

## 4. Statistics & Logging
*   **`tools.Statistics(key)`**: Compiles metrics (mean, std, min, max).
*   **`tools.MultiStatistics`**: Tracks multiple keys (e.g., Fitness AND Tree Size).
*   **`tools.Logbook`**: Stores stats per generation.
*   **`tools.HallOfFame(maxsize)`**: Persists the absolute best individuals found across the entire run (Elitism).
*   **`tools.History`**: Tracks genealogy (ancestry tree) of individuals.

## 5. Algorithms (`deap.algorithms`)
*   **`eaSimple`**: Generational (Evaluate -> Select -> Mate -> Mutate).
*   **`eaMuPlusLambda`**: Elitist (Next gen = Best of Parents + Offspring).
*   **`eaMuCommaLambda`**: Non-elitist (Next gen = Best of Offspring only).
*   **`eaGenerateUpdate`**: For manual generation control (e.g., external simulators).

## 6. Minimal Implementation Template

```python
import operator, random, numpy as np
from deap import algorithms, base, creator, tools, gp

def main():
    # 0. Reproducibility
    random.seed(42)

    # 1. SETUP: Typed Primitive Set
    pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addTerminal(1.0, float)
    pset.renameArguments(ARG0='x')

    # 2. CREATOR: FitnessMin (-1.0)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # 3. TOOLBOX
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evaluate(ind):
        # Compile tree to function
        func = toolbox.compile(expr=ind)
        # Vectorized check (assuming numpy input)
        # return tuple!
        return (abs(func(10.0) - 100.0),) 

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Critical: Bloat Control
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # 4. STATS
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    hof = tools.HallOfFame(1)

    # 5. RUN
    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=10, 
                        stats=stats, halloffame=hof, verbose=True)

if __name__ == "__main__":
    main()