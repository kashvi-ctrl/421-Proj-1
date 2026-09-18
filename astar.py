import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aima-python"))

from aima.search import Problem, astar_search

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from utils import route_cost

def mst_heuristic(matrix, unvisited_cities):
    """Returns the total cost of an MST connecting all cities in unvisited_cities."""
    if len(unvisited_cities) <= 1:
        return 0

    unvisited_list = list(unvisited_cities)
    sub_matrix = matrix[np.ix_(unvisited_list, unvisited_list)]

    mst = minimum_spanning_tree(sub_matrix)
    return mst.sum()

class TSP(Problem):
    def __init__(self, matrix, start=0):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.start = start

        initial = (start,)  
        super().__init__(initial)

    def actions(self, state):
        return list(set(range(self.n)) - set(state))

    def result(self, state, action):
        return state + (action,)

    def goal_test(self, state):
        return len(state) == self.n

    def path_cost(self, c, state1, action, state2):
        new_cost = c + self.matrix[state1[-1]][action]

        if self.goal_test(state2):
            new_cost += self.matrix[state2[-1]][self.start]

        return new_cost

    def h(self, node):
        unvisited = set(range(self.n)) - set(node.state)

        return mst_heuristic(self.matrix, unvisited)
    
if __name__ == "__main__":
    from utils import load_matrix
    from time import time

    matrix = load_matrix("data/10_random_adj_mat_0.txt")
    tsp = TSP(matrix, start=0)

    print("Running A*...")
    t0 = time()
    solution_node = astar_search(tsp)
    print("Solved in %f seconds" % (time() - t0))
    print("Route:", solution_node.state)
    print("Cost:", solution_node.path_cost)