from utils import route_cost
import random

def nearest_neighbor(matrix, start=0):
    n = matrix.shape[0]
    route = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    curr = start

    while unvisited:
        next = None
        best_dist = None

        for city in unvisited:
            dist = matrix[curr][city]
            if best_dist is None or dist < best_dist:
                best_dist = dist
                next = city

        route.append(next)
        unvisited.discard(next)
        curr = next

    return route


def helper(matrix, route):
    n = len(route)

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            new_route = route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]

            if route_cost(matrix, new_route) < route_cost(matrix, route):
                return new_route

    return None


def nearest_neighbor_2opt(matrix, start=0):
    route = nearest_neighbor(matrix, start)

    while True:
        new_route = helper(matrix, route)
        if new_route is None:
            break
        route = new_route

    return route


def random_nearest_neighbor(matrix, k, start=0):
    n = matrix.shape[0]
    route = [start]
    unvisited = set(range(n))
    unvisited.discard(start)
    curr = start

    while unvisited:
        sorted_by_dist = sorted(unvisited, key=lambda city: matrix[curr][city])
        candidates = sorted_by_dist[:k]
        next_city = random.choice(candidates)

        route.append(next_city)
        unvisited.discard(next_city)
        curr = next_city

    return route


def rrnn(matrix, k, num_repeats, start=0):
    best_route = None
    best_cost = None

    for _ in range(num_repeats):
        candidate_route = random_nearest_neighbor(matrix, k, start)

        while True:
            improved_route = helper(matrix, candidate_route)
            if improved_route is None:
                break
            candidate_route = improved_route

        candidate_cost = route_cost(matrix, candidate_route)

        if best_cost is None or candidate_cost < best_cost:
            best_cost = candidate_cost
            best_route = candidate_route

    return best_route


if __name__ == "__main__":
    import numpy as np

    example = np.array([
        [0, 20, 42, 35],
        [20, 0, 30, 34],
        [42, 30, 0, 12],
        [35, 34, 12, 0],
    ])
    print(nearest_neighbor(example, start=0))

if __name__ == "__main__":
    from utils import load_matrix, route_cost

    matrix = load_matrix("data/10_random_adj_mat_0.txt") 

    nn_route = nearest_neighbor(matrix, start=0)
    nn2o_route = nearest_neighbor_2opt(matrix, start=0)

    print("NN route:", nn_route)
    print("NN cost:", route_cost(matrix, nn_route))

    print("NN2Opt route:", nn2o_route)
    print("NN2Opt cost:", route_cost(matrix, nn2o_route))

if __name__ == "__main__":
    from utils import load_matrix

    matrix = load_matrix("data/10_random_adj_mat_0.txt")

    nn_route = nearest_neighbor(matrix, start=0)
    nn2o_route = nearest_neighbor_2opt(matrix, start=0)
    rrnn_route = rrnn(matrix, k=3, num_repeats=10, start=0)

    print("NN cost:", route_cost(matrix, nn_route))
    print("NN2Opt cost:", route_cost(matrix, nn2o_route))
    print("RRNN cost:", route_cost(matrix, rrnn_route))