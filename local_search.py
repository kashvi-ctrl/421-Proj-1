import random
import math
from utils import route_cost

def random_solution(n):
    return random.sample(range(n), n)


def hill_climbing(matrix, num_restarts, num_attempts=1000):
    n = matrix.shape[0]
    best_route = None
    best_cost = None

    for _ in range(num_restarts):
        route = random_solution(n)

        for _ in range(num_attempts):
            i, j = random.sample(range(n), 2)

            new_route = route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]

            if route_cost(matrix, new_route) < route_cost(matrix, route):
                route = new_route

        cost = route_cost(matrix, route)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route

def simulated_annealing(matrix, alpha, initial_temperature, max_iterations):
    n = matrix.shape[0]
    route = random_solution(n)
    cost = route_cost(matrix, route)

    best_route = route[:]
    best_cost = cost

    t = initial_temperature

    for _ in range(max_iterations):
        i, j = random.sample(range(n), 2)

        new_route = route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = route_cost(matrix, new_route)

        delta = cost - new_cost  # negative means new_cost is worse
        acceptance_probability = math.exp(delta / t)

        if new_cost < cost or random.random() < acceptance_probability:
            route = new_route
            cost = new_cost
            t = t * alpha

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route

import random
from utils import route_cost


def initial_population(n, population_size):
    return [random_solution(n) for _ in range(population_size)]


def crossover(parent1, parent2):
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))

    child = [None] * n
    child[start:end] = parent1[start:end]

    fill_values = [city for city in parent2 if city not in child]
    fill_positions = [i for i in range(n) if child[i] is None]

    for pos, city in zip(fill_positions, fill_values):
        child[pos] = city

    return child


def mutate(route, mutation_chance):
    if random.random() < mutation_chance:
        i, j = random.sample(range(len(route)), 2)
        route = route[:]
        route[i], route[j] = route[j], route[i]

    return route


def genetic_algorithm(matrix, mutation_chance, population_size, num_generations):
    n = matrix.shape[0]
    population = initial_population(n, population_size)

    def cost_of(route):
        return route_cost(matrix, route)

    for _ in range(num_generations):
        population.sort(key=cost_of)

        children = []
        while len(children) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_chance)
            children.append(child)

        combined = population + children
        combined.sort(key=cost_of)
        population = combined[:population_size]

    best_route = population[0]
    return best_route

if __name__ == "__main__":
    from utils import load_matrix

    matrix = load_matrix("data/10_random_adj_mat_0.txt")
    route = genetic_algorithm(matrix, mutation_chance=0.1, population_size=50, num_generations=200)
    print("Genetic Algorithm cost:", route_cost(matrix, route))