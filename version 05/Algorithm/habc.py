
import copy
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime
from bee import Bee

class Hybrid_Artificial_Bee_Colony:

    def __init__(self, run_id, category, problem_num, nK, nI, Capacity, Profits, Weights, cpu_time_limit, employed_bees_num, onlooker_bees_num, \
        max_try_improve, crossover_type, pc_onePoint, pc_uniForm, crossover_neighbor_selection, cn_k_tournoment, pm, \
        make_feasible_check, onlooker_selection, onlooker_k_tournoment, p_scout_delete, heuristic_phase_check, heuristic_iteration_num, \
        heuristic_selection, h_k_tournoment, p_set_zero, result_file_name):
        
        self.run_id = run_id
        self.category = category
        self.problem_num = problem_num
        self.cpu_time_limit = cpu_time_limit
        self.employed_bees_num = employed_bees_num
        self.nK = nK
        self.nI = nI
        self.capacity = Capacity
        self.profits = np.array(Profits)
        self.weights = np.array(Weights)
        self.onlooker_bees_num = onlooker_bees_num
        self.max_try_improve = max_try_improve
        self.pc_onePoint = pc_onePoint
        self.pc_uniForm = pc_uniForm
        self.pm = pm
        self.crossover_type = crossover_type
        self.result_file_name = result_file_name
        self.p_set_zero = p_set_zero
        self.heuristic_iteration_num = heuristic_iteration_num
        self.make_feasible_check = make_feasible_check
        self.p_scout_delete = p_scout_delete
        self.crossover_neighbor_selection = crossover_neighbor_selection
        self.cn_k_tournoment = cn_k_tournoment
        self.onlooker_selection = onlooker_selection
        self.onlooker_k_tournoment = onlooker_k_tournoment
        self.heuristic_phase_check = heuristic_phase_check
        self.heuristic_selection = heuristic_selection
        self.h_k_tournoment = h_k_tournoment
        
        self.bees = []

    def optimize(self):
        self.initialize_population()

        best_fitnesses_of_iteration = []
        best_fitnesses_so_far = []
        best_fitness_so_far = 0
        best_bee_so_far = Bee
        last_best_fitness_of_itration = 0
        time_number_list = []
    
        iteration_num = 0
        best_fitness_iteration_num = iteration_num # getting the last iteration that fitness ever updated

        currentTime = datetime.now().strftime("%H:%M:%S")
        print(f"run_id -> {self.run_id}: {currentTime}")
    
        start_t = time.time() # the start time of algorithm
        end_t = time.time()
        end_t_best_so_far = time.time()
        elapsed_time = 0

        while(elapsed_time<self.cpu_time_limit):

            result = open(self.result_file_name, 'a')    
            iteration_num+=1
            
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            if(self.heuristic_phase_check == 'yes'):
                self.heuristic_phase()
            
            best_bee_of_iteration, best_fitness_of_iteration = self.find_best_bee()

            if(best_fitness_of_iteration > best_fitness_so_far):
                best_fitness_so_far = best_fitness_of_iteration
                best_bee_so_far = best_bee_of_iteration
                print(f"best fitness so far: {best_fitness_so_far}")
                best_fitness_iteration_num = iteration_num
                end_t_best_so_far = time.time()
              
            if (last_best_fitness_of_itration != best_fitness_of_iteration):
                end_t = time.time()
                elapsed_best_updated = end_t - start_t
                time_number_list.append(elapsed_best_updated)
                best_fitnesses_of_iteration.append(best_fitness_of_iteration)
                best_fitnesses_so_far.append(best_fitness_so_far)                
                last_best_fitness_of_itration = best_fitness_of_iteration  
                
            self.scout_bees_phase()            

            current_time = time.time()
            elapsed_time = current_time-start_t # calculating the total time for checking limitation
                            
            result.close()    

        best_fitness_time = end_t_best_so_far - start_t

        return best_fitnesses_of_iteration, best_fitnesses_so_far, best_bee_so_far, best_fitness_so_far, \
            iteration_num, best_fitness_iteration_num, best_fitness_time, time_number_list

    def initialize_population(self):
        # making each random solution -> employed bees
        # each random solution is made by randomly choose answers, and make them 1, until it stays feasible

        for _ in range(self.employed_bees_num):
            bee = self.make_a_bee()
            self.bees.append(bee)

    def employed_bees_phase(self):
          
        # we try for improvement one time for each bee, if change happens we add one to improvement-try property of that bee
        for bee in self.bees:
            change_flag = self.try_for_improvement(bee)
            
            if(change_flag == False): 
                bee.try_improve += 1

    def onlooker_bees_phase(self):
        # by roulette wheel or tournoment we do cross_over and mutation for "onlooker_bees_num" times,
        
        # checking the precedure
        if (self.onlooker_selection == "Roulette Wheel"):
            selection_type = self.roulette_wheel()
        elif(self.onlooker_selection == "Tournoment"):
            selection_type = self.tournoment('onlooker')
        
        for _ in range(self.onlooker_bees_num):
            
            bee = selection_type
                            
            # we try for improvement one time for each bee, if change happens we add one to improvement-try property of that bee
            change_flag = self.try_for_improvement(bee)
            if(change_flag == False): 
                bee.try_improve += 1
                                                            
    def scout_bees_phase(self):
        # we delete the bees that have "try_improve" more than "max_try_improve", in percent of "p_scout_delete"

        limit_amount = self.p_scout_delete * self.employed_bees_num 
        amount = 0
        index = 0

        while index < len(self.bees) and amount<=limit_amount:
            bee = self.bees[index]
            if bee.try_improve >= self.max_try_improve:
                self.bees.pop(index)
                self.bees.append(self.make_a_bee())
                amount+=1
            index += 1

    def make_a_bee(self):
        # making each random solution -> employed bees
        # each random solution is made by randomly choose answers, and make them 1, until it stays feasible

        bee_main = Bee(self.nI)
        bee_secondary = Bee(self.nI)
        capacity_flag = True

        while(capacity_flag):
            x = random.randint(0, self.nI-1)
            if(bee_secondary.data[x]==0):
                bee_secondary.data[x] = 1
                capacity_flag = self.check_feasiblity(bee_secondary)
                if(capacity_flag):
                    bee_main.data[x] = 1
        
        self.calculate_fitness(bee_main)

        return bee_main

    def check_feasiblity(self, bee):
        # checking feasiblity of the answers that has been made (capacity)
        
        # Now perform the element-wise multiplication
        used_capacities = np.sum(bee.data * self.weights, axis=1)

        return np.all(used_capacities <= self.capacity)

    def try_for_improvement(self, bee):
        # we do the cross over and mutation here
        # we also return that if the process made any changes or not
        
        change_flag = False
        new_bee = copy.deepcopy(bee)
        
        # doing the cross over on selected bee and a neighbor (that will be handled in _cross_over)
        if(self.crossover_type == "one_point"):
            self.crossover_one_point(new_bee)
        elif(self.crossover_type == "uniform"):
            self.crossover_uniform(new_bee)
        
        # doing the mutation on selected bee
        self.mutation(new_bee)         
        
        # in here first we check the "make_feasible_check" parameter, if it was "yes", it means if our answer was not feasible,
        # we won't trash it away, we first make it feasible, then we checkthe improvement
        # but if "make_feasible_check" parameter be false it means we are going to continue like classic version
        if(self.make_feasible_check == 'yes'):
            if(self.check_feasiblity(new_bee)==False):
                self.make_feasible(new_bee)
                
            self.calculate_fitness(new_bee)
            if(new_bee.fitness > bee.fitness):
                change_flag = True     
                bee.data = new_bee.data.copy()
                bee.fitness = new_bee.fitness
                bee.try_improve = 0
        else:
            if(self.check_feasiblity(new_bee)):                
                self.calculate_fitness(new_bee)
                if(new_bee.fitness > bee.fitness):
                    change_flag = True     
                    bee.data = new_bee.data.copy()
                    bee.fitness = new_bee.fitness
                    bee.try_improve = 0

        return change_flag    

    def heuristic_phase(self): 
        # in here in amount of "heuristic_iteration_num" parameter, we do "set_one_to_zero" and "set_zero_to_one"
                
        if (self.heuristic_selection == "Roulette Wheel"):
            bee = self.roulette_wheel()
        elif(self.heuristic_selection == "Tournoment"):
            bee = self.tournoment('onlooker')
        
        for iter in range(self.heuristic_iteration_num):
            # with a probablity
            self.set_one_to_zero(bee)
            # until it stays feasible
            self.set_zero_to_one(bee)
                      
    def make_feasible(self, bee):
        # in here we get a infeasible answer we want to make it feasible
        # for a set probability named "set_to_zero_probability" we go through the answer(bee.data elements) and for each of them,
        # we check the probability and do the change, we go through this until we reach to a feasible answer
        
        feasiblity_flag = False
        while(feasiblity_flag==False):  
            self.set_one_to_zero(bee)
            feasiblity_flag = self.check_feasiblity(bee)
  
    def set_zero_to_one(self, bee):
        # in here we randomly choose a item and turn 0s, to 1s until it stays feasible
        feasiblity_flag = True
        demon_bee = copy.deepcopy(bee)
        
        while(feasiblity_flag):
            x = random.randint(0, self.nI-1)
            if(demon_bee.data[x]==0):
                demon_bee.data[x] = 1
                feasiblity_flag = self.check_feasiblity(demon_bee)
                if(feasiblity_flag):
                    bee.data[x] = 1
                            
    def set_one_to_zero(self, bee):
        # a list of random numbers  
        random_numbers = np.random.random(size=self.nI)
        # a boolean
        zero_mask = random_numbers <= self.p_set_zero
        # it checks where ever the random_number is equal or less than set_to_zero_probability, it set that index of bee.data to 0
        bee.data = np.where(zero_mask, 0, bee.data)

    def crossover_one_point(self, bee):
        # we get a bee as entry, firstly we check if the probablity would let us do the crossover on this bee or not
        # secondly if the probablity let us, we check the determine precedure
        # thirdly we choose a random position
        # at the end we change the entered bee.data somehow that the first part (before choosen position) be from itself, 
            # and the second part (after choosen position) be from neighbor bee
        
        # in here we do not neccesserly change the entered bee
                
        x = random.random()

        if(x<=self.pc_onePoint):
            
            if (self.crossover_neighbor_selection == 'Roulette Wheel'):
                neighbor_bee = self.roulette_wheel()
            elif (self.crossover_neighbor_selection == 'Tournoment'):
                neighbor_bee = self.tournoment('crossover')
            else:
                neighbor_bee = random.choice(self.bees)
                
            # choosing a random position for change
            random_pos = random.randint(1, self.nI-1)

            # in here we change parts of our "bee.data" base on choosed position,
            # the first part comes from bee.data, and the second part comes from neighbor.data 
            bee.data[random_pos:] = neighbor_bee.data[random_pos:].copy()

    def crossover_uniform(self, bee):
        # in here firstly we determine the precedure
        # secondly for each item in bee.data (entery) we check the cross_over probablity
        # we change the items that can pass the probablity, with the Adjacent item of neighbor bee
        
        # in here we mostly change the entered bee, but only a few items (it depends on "pc_uniForm")

        if (self.crossover_neighbor_selection == 'Roulette Wheel'):
            neighbor_bee = self.roulette_wheel()
        elif (self.crossover_neighbor_selection == 'Tournoment'):
            neighbor_bee = self.tournoment('crossover')
        else:
            neighbor_bee = random.choice(self.bees)

        # Generate a mask for crossover operation
        mask = np.random.rand(self.nI) <= self.pc_uniForm

        # Update the bee's data using the mask
        bee.data[mask] = neighbor_bee.data[mask].copy()
        
    def mutation(self, bee):
        # for each answer that employed bees have made, we select a random position and we change it with 0 or 1 (randomly)
        # only if the changed answer be better than the previous one and it be valid, it will change
        # we also return that if the muatation has done a change or not
        
        random_numbers = np.random.random(self.nI)
        mask = random_numbers <= self.pm
        bee.data[mask] = np.where(bee.data[mask] == 0, 1, 0)

    def calculate_fitness(self, bee):
        # fitness is amount of capacity that the bee can take (the capacity that the answer is occupying)
        bee.fitness = np.sum(self.profits * bee.data)

    def roulette_wheel(self):

        sum_of_fitnesses = sum([bee.fitness for bee in self.bees])

        rand_num = random.uniform(0, sum_of_fitnesses)
        cumulative_fitness = 0
        
        for bee in self.bees:
            cumulative_fitness += bee.fitness
            if cumulative_fitness >= rand_num:
                return bee
        
    def tournoment(self, caller_func):
        # choosing our bee with tournoment procedure with "k_tournoment" variable
        
        if(caller_func == 'onlooker'):
            k_tournoment = self.onlooker_k_tournoment
        elif(caller_func == 'crossover'):
            k_tournoment = self.cn_k_tournoment
        elif(caller_func == 'heuristic'):
            k_tournoment = self.h_k_tournoment
        
        bees_array = np.array(self.bees)  # Convert list of bees to numpy array

        # Randomly select k_tournament bees
        selected_indices = np.random.choice(len(bees_array), k_tournoment, replace=False)
        tournament_list = bees_array[selected_indices]

        # Find bee with maximum fitness
        max_bee_index = np.argmax([bee.fitness for bee in tournament_list])
        max_bee = tournament_list[max_bee_index]

        return max_bee

        # tournoment_list = []
        # for i in range(k_tournoment):
        #     tournoment_list.append(random.choice(self.bees))
        # max_F = 0
        # max_B = None
        # for bee in tournoment_list:
        #     if(bee.fitness>max_F):
        #         max_F = bee.fitness
        #         max_B = bee
        # return max_B
    
    def find_best_bee(self):
        fitness_values = np.zeros(len(self.bees))

        for i, bee in enumerate(self.bees):
            fitness_values[i] = bee.fitness

        best_index = np.argmax(fitness_values)
        best_bee = self.bees[best_index]
        best_fitness = fitness_values[best_index]

        return best_bee, best_fitness

    def write_results(self, best_fitnesses_of_iterations, best_final_bee, best_final_fitness, st):

        result = open(self.result_file_name, 'a')
        result.write("FINAL RESULT \n \n")
            
        fitness_avg = sum(best_fitnesses_of_iterations)/len(best_fitnesses_of_iterations)
        result.write(f"the best final Bee => \ndata: {best_final_bee.data}, fitness: {best_final_fitness} \n")
        result.write(f"the average fitness of all: {fitness_avg} \n \n")

        # end time of all
        et = time.time()

        elapsed_time = et - st
        result.write(f'Execution time of all: {elapsed_time} seconds \n \n')

        result.write("------------------------ \n")
        result.write("COMPARE ANSWER \n \n")
        result.write(f"real answer = \n")
        result.write(f"my answer = {best_final_fitness} \n")

        result.write("------------------------ \n")
        result.write("PARAMETERS \n \n")
        result.write(f"run_id = {self.run_id}\n")
        result.write(f"category = {self.category}\n")
        result.write(f"problem_num = {self.problem_num}\n")
        result.write(f"cpu_time_limit = {self.cpu_time_limit}\n")
        result.write(f"employed_bees_num = {self.employed_bees_num}\n")
        result.write(f"onlooker_bees_num = {self.onlooker_bees_num}\n")
        result.write(f"max_try_improve = {self.max_try_improve}\n")
        result.write(f"crossover_type = {self.crossover_type}\n")
        result.write(f"pc_onePoint = {self.pc_onePoint}\n")
        result.write(f"pc_uniForm = {self.pc_uniForm}\n")
        result.write(f"crossover_neighbor_selection = {self.crossover_neighbor_selection}\n")
        result.write(f"cn_k_tournoment = {self.cn_k_tournoment}\n")
        result.write(f"pm = {self.pm}\n")
        result.write(f"make_feasible_check = {self.make_feasible_check}\n")
        result.write(f"onlooker_selection = {self.onlooker_selection}\n")
        result.write(f"onlooker_k_tournoment = {self.onlooker_k_tournoment}\n")
        result.write(f"p_scout_delete = {self.p_scout_delete}\n")
        result.write(f"heuristic_phase_check = {self.heuristic_phase_check}\n")
        result.write(f"heuristic_iteration_num = {self.heuristic_iteration_num}\n")
        result.write(f"heuristic_selection = {self.heuristic_selection}\n")
        result.write(f"h_k_tournoment = {self.h_k_tournoment}\n")
        result.write(f"p_set_zero = {self.p_set_zero}\n")

        result.close()

    def write_excel(self, run_id, category, problem_num, cpu_time_limit, employed_bees_num, onlooker_bees_num, max_try_improve, \
        crossover_type, pc_onePoint, pc_uniForm, crossover_neighbor_selection, cn_k_tournoment, pm, make_feasible_check, \
        onlooker_selection, onlooker_k_tournoment, p_scout_delete, heuristic_phase_check, heuristic_iteration_num, heuristic_selection, \
        h_k_tournoment, p_set_zero, best_final_fitness, best_fitness_iteration_num, best_fitness_time, total_iteration_num):
        
        file_path = 'input_output/output.xlsx'

        df = pd.read_excel(file_path)
        
        new_data = {'run_id': run_id,
                    'category': category,
                    'problem_num': problem_num,
                    'cpu_time_limit': cpu_time_limit,
                    'employed_bees_num': employed_bees_num,
                    'onlooker_bees_num': onlooker_bees_num,
                    'max_try_improve': max_try_improve,
                    'crossover_type': crossover_type,
                    'pc_onePoint': pc_onePoint,
                    'pc_uniForm': pc_uniForm,
                    'crossover_neighbor_selection': crossover_neighbor_selection,
                    'cn_k_tournoment': cn_k_tournoment,
                    'pm': pm,
                    'make_feasible_check': make_feasible_check,
                    'onlooker_selection': onlooker_selection,
                    'onlooker_k_tournoment': onlooker_k_tournoment,
                    'p_scout_delete': p_scout_delete, 
                    'heuristic_phase_check': heuristic_phase_check,
                    'heuristic_iteration_num': heuristic_iteration_num,
                    'heuristic_selection': heuristic_selection,
                    'h_k_tournoment': h_k_tournoment,
                    'p_set_zero': p_set_zero,
                    'best_final_fitness': best_final_fitness,
                    'best_fitness_iteration_num': best_fitness_iteration_num,
                    'best_fitness_time': best_fitness_time,
                    'total_iteration_num': total_iteration_num
                    }
        
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_excel(file_path, index=False)