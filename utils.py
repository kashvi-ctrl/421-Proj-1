import time
import numpy as np

def load_matrix(fname):
    return np.loadtxt(fname)

def route_cost(matrix, route):
    cost = 0.0
    n = len(route)
    for i in range(n):
        a = route[i]
        b = route[(i + 1) % n]  # wraps back to route[0] on the last step
        cost += matrix[a][b]
    return cost

def time_function(func, *args, repeats=1, **kwargs):
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
    costs = [t[0] for t in triples]
    cpu_times = [t[1] for t in triples]
    real_times = [t[2] for t in triples]

    return (
        float(np.median(costs)),
        float(np.median(cpu_times)),
        float(np.median(real_times)),
    )
