import numpy as np

class Bee:
    def __init__(self, nI):
        self.data = np.zeros(nI)
        self.fitness = 0
        self.try_improve = 0