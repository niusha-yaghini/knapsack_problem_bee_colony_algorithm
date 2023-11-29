import Artificial_Bee_Colony
import numpy as np
import Reading_Data
import time
import copy
import Diagram
from datetime import datetime

def Bee_Colony_Algorithm():
    
    population = []
        
    best_bees_of_each_inner_iteration = []
    best_fitnesses_of_each_inner_iteration = []
    best_fitnesses_so_far = []
    for i in range(inner_iteration_of_algorithm):  

        # iteration_st = time.time()  # start time of iteration
        result = open(f'{result_file_name}', 'a')    
        # result.write(f"iteration number {i}: \n")
        currentTime = datetime.now().strftime("%H:%M:%S")
        print(f"iteration number {i}: {currentTime}")
 
        ABC = Artificial_Bee_Colony.ABC_algorithm(employed_bees_num, nK, nI, Capacity, Profits, Weights, onlooker_bees_num, max_improvement_try, pc, pm, k_tournomet_percent)
        ABC.employed_bees(population)
        ABC.onlooker_bees(population)
        best_bee_of_iteration, best_fitness_of_iteration = ABC.finding_best_bee(population)
        
        best_bees_of_each_inner_iteration.append(copy.deepcopy(best_bee_of_iteration))
        best_fitnesses_of_each_inner_iteration.append(copy.deepcopy(best_fitness_of_iteration))
        best_fitness_so_far = max(best_fitnesses_of_each_inner_iteration)
        best_fitnesses_so_far.append(best_fitness_so_far)

        # result.write(f"best bee => data: {best_bee_of_iteration.data}, fitness: {best_fitness_of_iteration}\n")  
        # result.write(f"best fitness so far: {best_fitness_so_far}\n")

        print(f"best fitness of iteration = {best_fitness_of_iteration}")
        print(f"best fitness so far: {best_fitness_so_far}")

        ABC.scout_bees(population)
        
        # iteration_et = time.time()  # end time of iteration
        # iteration_elapsed_time = iteration_et - iteration_st
        # result.write(f"Execution time of iteration: {iteration_elapsed_time} seconds\n \n")
        
        result.close()
    
    return best_bees_of_each_inner_iteration, best_fitnesses_of_each_inner_iteration, best_fitnesses_so_far


if __name__ == '__main__':
    
    employed_bees_num = 1000  # number of total bees => npop/2 = amount of first population
                         # this must be an even number 
    onlooker_bees_num = 100   # number of iterations in roulette wheel, that select a bee and pass it to improvement-try
    max_improvement_try = 50
    inner_iteration_of_algorithm = 1000
    pc = 0.7 # the probblity of cross-over
    pm = 2 # the probblity of mutation (pm/items)
    k_tournomet_percent = 0.1 # in amount of "k_tournomet/items", tournoment will choose, and return the best of them
    # percedure_type = "Tournoment"
    percedure_type = "Roulette Wheel"
    cross_over_type = "one_point"
    
    # file name of the datas
    data_file_name = ".\\mknap1-Question\\07.txt"
    
    # file name for save results
    result_file_name = ".\\mknap1-Answer(try6)\\07-4.txt"
    photo_name = "07-4"

    # nK = number of knapstacks
    # nI = number of items
    nK, nI, Capacity, Profits, Weights, real_answer = Reading_Data.Reading(data_file_name)
    
    # getting result by bees :)
    
    result = open(f'{result_file_name}', 'a')
    result.write(f"Artificial Bee Colony Algorithm \n \n")        
    result.close()

    # 1) writing the results in a text
    # 2) getting the time of algorithm in each iteration
    
    st = time.time() # get the start time of all     
       
    best_bees_of_iterations, best_fitnesses_of_iterations, best_fitnesses_so_far = Bee_Colony_Algorithm()

    # clearfiying the bee
    best_final_fitness = max(best_fitnesses_so_far)
    best_final_bee = None
    for b in best_bees_of_iterations:
        if(b.fitness == best_final_fitness):
            best_final_bee = b
            
    # writing the result
    result = open(f'{result_file_name}', 'a')
    # result.write("------------------------ \n")
    result.write("FINAL RESULT \n \n")
        
    # fitness_avg = np.average(best_fitnesses_of_iterations)
    fitness_avg = sum(best_fitnesses_of_iterations)/len(best_fitnesses_of_iterations)
    result.write(f"the best final Bee => \ndata: {best_final_bee.data}, fitness: {best_final_fitness} \n")
    result.write(f"the average fitness of all: {fitness_avg} \n \n")

    # end time of all
    et = time.time()

    elapsed_time = et - st
    result.write(f'Execution time of all: {elapsed_time} seconds \n \n')

    result.write("------------------------ \n")
    result.write("COMPARE ANSWER \n \n")
    result.write(f"real answer = {real_answer}\n")
    result.write(f"my answer = {best_final_fitness} \n")
    gap = real_answer - best_final_fitness
    result.write(f"gap = {gap}\n")
    gap_percent = (gap/real_answer)*100
    result.write(f"gap percent = {gap_percent}\n \n")
    # result.write("try4 gap = \n")
    # result.write("betterment than try4 = \n \n")    

    result.write("------------------------ \n")
    result.write("PARAMETERS \n \n")
    result.write(f"Number of Employed Bees = {employed_bees_num}\n")
    result.write(f"Number of Onlooker Bees = {onlooker_bees_num}\n")
    result.write(f"Max improvement try = {max_improvement_try}\n")
    result.write(f"Number of ABC algorithm's iterations = {inner_iteration_of_algorithm}\n")
    result.write(f"cross_over type = {cross_over_type}\n")
    result.write(f"Probblity of cross-over = {pc}\n")
    result.write(f"Probblity of mutation = {pm}\n")
    result.write(f"K tournoment percent = {k_tournomet_percent}\n")
    result.write(f"Precedure Type = {percedure_type}")

    print("---------------------------------")
    print(f"the best fitness of all: {best_final_fitness}")

    result.close()
    
    iteration_number_list = [i for i in range(1, inner_iteration_of_algorithm+1)]
    Diagram.diagram(iteration_number_list, best_fitnesses_of_iterations, best_fitnesses_so_far, photo_name)
    
    print()