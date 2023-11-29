import numpy as np
import Bees
import random
import copy

class ABC_algorithm():
    # artificial bee colony algorithm 

    def __init__(self, population_num, nK, nI, Capacity, Profits, Weights, onlooker_bees_num, Max_imporovement_try, pc, pm, k_tournomet_percent):
        self.employed_bees_num = population_num
        self.knapsacks = nK
        self.items = nI
        self.capacity = Capacity
        self.profits = Profits
        self.weights = Weights
        self.onlooker_bees_num = onlooker_bees_num
        self.Max_imporovement_try = Max_imporovement_try
        self.crossover_probbility = pc
        self.mutation_probblity = pm/self.items
        self.k_tournoment = int(k_tournomet_percent*self.items)
          
    def employed_bees(self, population):
        # making initial random answers (equal to amount of employed bees number)
        # do the improvement-try once on each of them
        # return the made answers
        
        if(len(population) == 0):
            for i in range(self.employed_bees_num):
                bee = self._making_bee()
                population.append(bee)
            
        # we try for improvement one time for each bee, if change happens we add one to improvement-try property of that bee
        for bee in population:
            change_flag = self._try_for_improvement(population, bee)
            if(change_flag): 
                bee.improvement_try = 0
                Bees.Bee._calculating_fitness(bee, self.items, self.profits)
            else: 
                bee.improvement_try += 1
                    
    def _making_bee(self):
        # making each random solution -> employed bees
        
        capacity_flag = True
        bee = Bees.Bee(self.items)
        new_bee = copy.deepcopy(bee)
        while(capacity_flag):
            x = random.randint(0, self.items-1)
            if(new_bee.data[x]==0):
                new_bee.data[x] = 1
                capacity_flag = self._validality_check(new_bee)
                if(capacity_flag):
                    bee.data[x] = 1
                
        return bee
                
    def _validality_check(self, bee):
        # checking validality of the answers that has been made (capacity)
        
        for j in range(self.knapsacks):
            current_capacity = self.capacity[j]
            my_capacity = 0
            for i in range(self.items):
                if (bee.data[i]==1):
                    my_capacity += self.weights[j][i]
            if(my_capacity>current_capacity):
                return False
        return True
                
    def onlooker_bees(self, population):
        # by rolette wheel precedure we do "onlooker_bees_num" times cross_over and mutation,
        # on solution that employed bees have made
                
        for bee in population:
            if(bee.fitness == None):
                Bees.Bee._calculating_fitness(bee, self.items, self.profits)
        
        sum_of_fitnesses = sum([bee.fitness for bee in population])
        
        for i in range(self.onlooker_bees_num):
            
            # # selecting the bee by roulette wheel
            # bee = self._roulette_wheel(population, sum_of_fitnesses)
            
            # sele a bee by tournoment procedure
            bee = self._tournoment(population)
            
            # we try for improvement one time for each bee, if change happens we add one to improvement-try property of that bee
            change_flag = self._try_for_improvement(population, bee)
            if(change_flag): 
                bee.improvement_try = 0
                Bees.Bee._calculating_fitness(bee, self.items, self.profits)
            else: 
                bee.improvement_try += 1
                                                        
    def scout_bees(self, population):
        delete_bees = []
        new_bees = []
        for bee in population:
            if(bee.improvement_try>=self.Max_imporovement_try):
                delete_bees.append(bee)
                new_bees.append(self._making_bee())
        for i in range(len(delete_bees)):
            population.remove(delete_bees[i])
            population.append(new_bees[i])
                    
    def _try_for_improvement(self, population, bee):
        # we do the cross over and mutation here
        # we also return that if the process made any changes or not
        
        change_flag = False
        new_bee = copy.deepcopy(bee)
        
        # doing the cross over on selected bee and a neighbor (that will be handled in _cross_over)
        self._cross_over_one_point(population, new_bee)
        
        # doing the mutation on selected bee
        self._mutation(new_bee) 
        
        if(self._validality_check(new_bee) and self._improvement_check(bee, new_bee)):
            bee.data = new_bee.data
            change_flag = True

        return change_flag        
    
    def _tournoment(self, population):
        tournoment_list = []
        for i in range(self.k_tournoment):
            tournoment_list.append(random.choice(population))
            
        maxF = 0
        max_B = None
        for bee in population:
            if(bee.fitness>maxF):
                maxF = bee.fitness
                max_B = bee
        return max_B
    
    def _roulette_wheel(self, population, sum_of_fitnesses):
        
        # choose a random number for selecting our bee    
        pick = random.uniform(0, sum_of_fitnesses)
        
        # selecting our bee by the "pick" number and roulette wheel procedure
        current = 0
        for bee in population:
            current += bee.fitness
            if current >= pick:
                return bee         
                
    def _cross_over_one_point(self, population, bee):
        # for each answer that employed bees have made, we select a radom neighbor
        # for each answer we also select a random position, and it replaced with its neighbors pos
        # if the changed answer be better than the previous one and it be valid, it will change
        # we also return that if the cross-over has done a change or not
        
        x = random.random()

        if(x<=self.crossover_probbility):
            # choosing a random position for change
            random_pos = random.randint(2, self.items-1)
            
            # choosing a neighbor, and it does not matter if it is the bee itself
            neighbor_bee = random.choice(population)
            
            self.replace_terms(bee, neighbor_bee, random_pos)
        
    def replace_terms(self, bee, neighbor_bee, random_pos):
        # in here we change parts of our choromosome base on choosed term
        
        data = []
        for i in range(random_pos):
            data.append(bee.data[i])
        for j in range(random_pos, self.items):
            data.append(neighbor_bee.data[j])
        
        bee.data = data
                
    def _cross_over(self, population, bee):
        # for each answer that employed bees have made, we select a radom neighbor
        # for each answer we also select a random position, and it replaced with its neighbors pos
        # if the changed answer be better than the previous one and it be valid, it will change
        # we also return that if the cross-over has done a change or not
        
        x = random.random()

        if(x<=self.crossover_probbility):
            # choosing a random position for change
            random_pos = random.randint(0, self.items-1)
            
            # choosing a neighbor, and it does not matter if it is the bee itself
            random_neighbor = random.choice(population)
        
            # checking that if the two position of bees are different or not (if they were different we do the replacement)
            if(bee.data[random_pos] != random_neighbor.data[random_pos]):
                bee.data[random_pos] = random_neighbor.data[random_pos]
                                            
    def _mutation(self, bee):
        # for each answer that employed bees have made, we select a random position and we change it with 0 or 1 (randomly)
        # only if the changed answer be better than the previous one and it be valid, it will change
        # we also return that if the muatation has done a change or not
        
        for i in range(self.items):            
            x = random.random()
            if(x<=self.mutation_probblity):
                bee.data[i] = 1 if bee.data[i] == 0 else 0
                
    def _improvement_check(self, current_bee, new_bee):
        # checking that the new bee (changed bee by cross_over or mutation) has imporoved or not
        
        Bees.Bee._calculating_fitness(current_bee, self.items, self.profits)
        Bees.Bee._calculating_fitness(new_bee, self.items, self.profits)
        return True if new_bee.fitness>current_bee.fitness else False
    
    def finding_best_bee(self, population):
        # finding the best solution
        
        best_fitness = 0
        best_bee = None
        for bee in population:
            Bees.Bee._calculating_fitness(bee, self.items, self.profits)
            if(bee.fitness>best_fitness):
                best_fitness = bee.fitness
                best_bee = bee

        return best_bee, best_fitness