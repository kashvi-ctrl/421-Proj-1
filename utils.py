"""
Shared utilities for CMSC 421 Project 1 (TSP).

Import this in part1_greedy.py, part2_astar.py, and part3_local_search.py
so you don't duplicate the loading/timing/cost logic in every file.
"""

import time
import numpy as np


def load_matrix(fname):
    """
    Load an adjacency matrix from a file where rows are newline-delimited
    and columns are space-delimited (see Section 1.2 of the project spec).
    """
    return np.loadtxt(fname)


def route_cost(matrix, route):
    """
    Given an adjacency matrix and a route (list of city indices, e.g.
    [0, 3, 1, 2]), return the total cost of the tour, including the
    edge back to the starting city.
    """
    cost = 0.0
    n = len(route)
    for i in range(n):
        a = route[i]
        b = route[(i + 1) % n]  # wraps back to route[0] on the last step
        cost += matrix[a][b]
    return cost


def time_function(func, *args, repeats=1, **kwargs):
    """
    Runs `func(*args, **kwargs)` `repeats` times and returns:
        (result, median_real_time_ns, median_cpu_time_ns)

    result is the output from the LAST call.

    Use `repeats` > 1 if a single call's CPU time reports as 0 (the spec
    calls this out in Section 2.2) — this divides total time across
    multiple runs instead.

    NOTE: for algorithms with randomness (NN's random start, RRNN, HC,
    SA, GA), don't confuse this `repeats` (for timing stability) with
    the R from Section 1.4 (for computing median cost/time across runs).
    You'll likely want to call this function once per R-run and collect
    the medians yourself at that level, rather than relying on this
    repeats param to do your R-run averaging for you.
    """
    real_times = []
    cpu_times = []
    result = None

    for _ in range(repeats):
        t0_real = time.time_ns()
        t0_cpu = time.process_time_ns()

        result = func(*args, **kwargs)

        t1_real = time.time_ns()
        t1_cpu = time.process_time_ns()

        real_times.append(t1_real - t0_real)
        cpu_times.append(t1_cpu - t0_cpu)

    median_real = float(np.median(real_times))
    median_cpu = float(np.median(cpu_times))

    return result, median_real, median_cpu


def median_of_medians(triples):
    """
    Helper for Section 1.4's aggregation procedure.

    Given a list of (cost, cpu_time, real_time) triples (one triple per
    matrix, already reduced to a median across R runs), returns the
    median cost, median cpu_time, and median real_time across all
    matrices of that size.
    """
    costs = [t[0] for t in triples]
    cpu_times = [t[1] for t in triples]
    real_times = [t[2] for t in triples]

    return (
        float(np.median(costs)),
        float(np.median(cpu_times)),
        float(np.median(real_times)),
    )


if __name__ == "__main__":
    # quick sanity check using the Figure 1 example from the spec
    example = np.array([
        [0, 20, 42, 35],
        [20, 0, 30, 34],
        [42, 30, 0, 12],
        [35, 34, 12, 0],
    ])
    route = [0, 1, 3, 2]
    print("Example cost:", route_cost(example, route))  # should be 20+34+12+42 = 108
