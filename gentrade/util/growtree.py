import random
# import re
import sys
# import warnings

# from collections import defaultdict, deque
# from functools import partial, wraps
from inspect import isclass

def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth >= height

    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth >= height or \
               (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def add_terminal(pset, type_, expr):
    try:

        term = random.choice(pset.terminals[type_])
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError("The gp.generate function tried to add " \
                         "a terminal of type '%s', but there is " \
                         "none available." % (type_,)).with_traceback(traceback)
    if isclass(term):
        term = term()
    expr.append(term)


# def is_valid_terminal(type_, pset):
#     return len(pset.terminals[type_]) > 0
from deap import gp

# def generate(pset, min_, max_, condition, type_=None):
#     for _ in range(100):
#         try:
#             t = try_generate(pset, min_, max_, condition, type_=type_)
#             tree = gp.PrimitiveTree(t)
#         except IndexError as e:
#             pass
#         else:
#             if tree.height >= min_:
#                 return tree
#     raise Exception('Tree generation not possible')

def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):# and is_valid_terminal(type_, pset):
            add_terminal(pset, type_, expr)
        else:
            prims = pset.primitives[type_]
            if len(prims) > 0:
                prim = random.choice(prims)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            else:
                add_terminal(pset, type_, expr)

    return expr
