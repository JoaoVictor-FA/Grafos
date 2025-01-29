"""Perturbation schemes used in local search-based algorithms.
The functions here should receive a permutation list with numbers from 0 to `n`
and return a generator with all neighbors of this permutation.
The neighbors should preferably be randomized to be used in Simulated
Annealing, which samples a single neighbor at a time.

References
----------
.. [1] Goulart, Fillipe, et al. "Permutation-based optimization for the load
    restoration problem with improved time estimation of maneuvers."
    International Journal of Electrical Power & Energy Systems 101 (2018):
    339-355.
.. [2] 2-opt: https://en.wikipedia.org/wiki/2-opt
"""

from random import sample
from typing import Callable, Dict, Generator, List


def two_opt_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """2-opt perturbation scheme [2]"""
    n = len(x)
    i_range = range(2, n)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 1, n + 1)
        for j in sample(j_range, len(j_range)):
            xn = x.copy()
            xn = xn[: i - 1] + list(reversed(xn[i - 1 : j])) + xn[j:]
            yield xn


# Mapping with all possible neighborhood generators in this module
neighborhood_gen: Dict[
    str, Callable[[List[int]], Generator[List[int], List[int], None]]
] = {
    "two_opt": two_opt_gen,
}
