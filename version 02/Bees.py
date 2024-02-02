import numpy as np

class Bee:
    # each bee is equal to a valid solution, that has the body(data = array answer), and its fitness
    
    def __init__(self, items):
        self.data = np.zeros(items)
        self.fitness = None
        self.improvement_try = 0

    def _calculating_fitness(self, items, profits):
        # fitness is amount of capacity that the bee can take (the capacity that the answer is occupying)
        
        fitness = 0
        for i in range(items):
            if(self.data[i]==1):
                fitness += profits[i]
        self.fitness = fitness