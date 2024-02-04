import Artificial_Bee_Colony
import numpy as np
import Reading_Data_mknapcb
import Reading_Data_petersen
import time
import copy
import Diagram
from datetime import datetime

def Bee_Colony_Algorithm(nK, nI, Capacity, Profits, Weights, result_file_name):
    
    population = []
        
    best_bees_of_each_inner_iteration = []
    best_fitnesses_of_each_inner_iteration = []
    best_fitnesses_so_far = []
    
    # for i in range(inner_iteration_of_algorithm):  
    start_limitation_time = time.time()
    elapsed_limitation_time = 0
    
    iteration_num = 1
    while(elapsed_limitation_time<cpu_time_limit):

        iteration_st = time.time()  # start time of iteration
        result = open(result_file_name, 'a')    
        # result.write(f"iteration number {iteration_num}: \n")
        currentTime = datetime.now().strftime("%H:%M:%S")
        # print(f"iteration number {iteration_num}: {currentTime}")
        iteration_num+=1
 
        ABC = Artificial_Bee_Colony.ABC_algorithm(employed_bees_num, nK, nI, Capacity, Profits, Weights, onlooker_bees_num, max_improvement_try, pc, pm, k_tournomet_percent, percedure_type, p_setZero, heuristic_iteration_num)
        ABC.employed_bees(population)
        ABC.onlooker_bees(population)
        best_bee_of_iteration, best_fitness_of_iteration = ABC.finding_best_bee(population)
        
        best_bees_of_each_inner_iteration.append(copy.deepcopy(best_bee_of_iteration))
        best_fitnesses_of_each_inner_iteration.append(copy.deepcopy(best_fitness_of_iteration))
        best_fitness_so_far = max(best_fitnesses_of_each_inner_iteration)
        best_fitnesses_so_far.append(best_fitness_so_far)

        # result.write(f"best bee => data: {best_bee_of_iteration.data}, fitness: {best_fitness_of_iteration}\n")  
        # result.write(f"best fitness so far: {best_fitness_so_far}\n")

        # print(f"best fitness of iteration = {best_fitness_of_iteration}")
        print(f"best fitness so far: {best_fitness_so_far}")

        ABC.scout_bees(population)
        
        current_limitation_time = time.time()
        elapsed_limitation_time = current_limitation_time-start_limitation_time;
                
        iteration_et = time.time()  # end time of iteration
        iteration_elapsed_time = iteration_et - iteration_st
        # result.write(f"Execution time of iteration: {iteration_elapsed_time} seconds\n \n")
        
        result.close()
    
    return best_bees_of_each_inner_iteration, best_fitnesses_of_each_inner_iteration, best_fitnesses_so_far, iteration_num


def how_to_read(file_type):
    if(file_type=="petersen"):
        # name of files that needs to be read
        input_files_name = [".\\petersen\\mknap1-Question\\01.txt",
                            ".\\petersen\\mknap1-Question\\02.txt",
                            ".\\petersen\\mknap1-Question\\03.txt",
                            ".\\petersen\\mknap1-Question\\04.txt",
                            ".\\petersen\\mknap1-Question\\05.txt",
                            ".\\petersen\\mknap1-Question\\06.txt",
                            ".\\petersen\\mknap1-Question\\07.txt"]
        
        # name of files that datas would be written here
        output_files_name = [".\\petersen\\mknap1-Answer(try2)\\01.txt",
                             ".\\petersen\\mknap1-Answer(try2)\\02.txt",
                             ".\\petersen\\mknap1-Answer(try2)\\03.txt",
                             ".\\petersen\\mknap1-Answer(try2)\\04.txt",
                             ".\\petersen\\mknap1-Answer(try2)\\05.txt",
                             ".\\petersen\\mknap1-Answer(try2)\\06.txt",
                             ".\\petersen\\mknap1-Answer(try2)\\07.txt"]
        
        # name of photos that results would be shown in
        photos_name = ["petersen_01",
                       "petersen_02",
                       "petersen_03",
                       "petersen_04",
                       "petersen_05",
                       "petersen_06",
                       "petersen_07"]
        
        for file_num in range(len(input_files_name)):
            
            # nK = number of knapstacks
            # nI = number of items
            # nK, nI, Capacity, Profits, Weights = Reading_Data.Reading(data_file_name)
            nK, nI, Capacity, Profits, Weights, real_answer = Reading_Data_petersen.Reading(input_files_name[file_num])
            
            out_file = f'{output_files_name[file_num]}'
            result = open(out_file, 'a')
            result.write(f"Artificial Bee Colony Algorithm on knapsack\n \n")        
            result.close()

            # 1) writing the results in a text
            # 2) getting the time of algorithm in each iteration
            st = time.time() # get the start time of all     
            
            # getting result by bees :)
            best_bees_of_iterations, best_fitnesses_of_iterations, best_fitnesses_so_far, iteration_num = Bee_Colony_Algorithm(nK, nI, Capacity, Profits, Weights, out_file)

            # clearfiying the bee
            best_final_fitness = max(best_fitnesses_so_far)
            best_final_bee = None
            for b in best_bees_of_iterations:
                if(b.fitness == best_final_fitness):
                    best_final_bee = b
                    
            # writing the result
            result = open(out_file, 'a')
            # result.write("------------------------ \n")
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
            result.write(f"real answer = {real_answer}\n")
            result.write(f"my answer = {best_final_fitness} \n")
            gap = real_answer - best_final_fitness
            result.write(f"gap = {gap}\n")
            gap_percent = (gap/real_answer)*100
            result.write(f"gap percent = {gap_percent}\n \n")

            result.write("------------------------ \n")
            result.write("PARAMETERS \n \n")
            result.write(f"Number of Employed Bees = {employed_bees_num}\n")
            result.write(f"Number of Onlooker Bees = {onlooker_bees_num}\n")
            result.write(f"Max improvement try = {max_improvement_try}\n")
            result.write(f"cross_over type = {cross_over_type}\n")
            result.write(f"probability of cross-over = {pc}\n")
            result.write(f"probability of mutation = {pm}\n")
            result.write(f"K tournoment percent = {k_tournomet_percent}\n")
            result.write(f"Precedure Type = {percedure_type}\n")
            result.write(f"Number of ABC algorithm's iterations = {iteration_num}\n")
            result.write(f"Limited Time = {cpu_time_limit}")

            # print("---------------------------------")
            # print(f"the best fitness of all: {best_final_fitness}")

            result.close()
            
            iteration_number_list = [i for i in range(1, iteration_num)]
            Diagram.diagram(iteration_number_list, best_fitnesses_of_iterations, best_fitnesses_so_far, photos_name[file_num])
               
    elif(file_type=="mknapcb"):
        examples_amount = 3
        
        # name of files that needs to be read
        input_files_name = [".\\mknapcb\\01\\Question\\mknapcb1.txt",
                            ".\\mknapcb\\02\\Question\\mknapcb2.txt",
                            ".\\mknapcb\\03\\Question\\mknapcb3.txt",
                            ".\\mknapcb\\04\\Question\\mknapcb4.txt",
                            ".\\mknapcb\\05\\Question\\mknapcb5.txt",
                            ".\\mknapcb\\06\\Question\\mknapcb6.txt",
                            ".\\mknapcb\\07\\Question\\mknapcb7.txt",
                            ".\\mknapcb\\08\\Question\\mknapcb8.txt",
                            ".\\mknapcb\\09\\Question\\mknapcb9.txt"]
        
        # name of files that datas would be written here
        output_files_name = [".\\mknapcb\\01\\Answer",
                             ".\\mknapcb\\02\\Answer",
                             ".\\mknapcb\\03\\Answer",
                             ".\\mknapcb\\04\\Answer",
                             ".\\mknapcb\\05\\Answer",
                             ".\\mknapcb\\06\\Answer",
                             ".\\mknapcb\\07\\Answer",
                             ".\\mknapcb\\08\\Answer",
                             ".\\mknapcb\\09\\Answer"]
         
        # name of photos that results would be shown in
        photos_name = ["mknapcb1_0",
                       "mknapcb2_0",
                       "mknapcb3_0",
                       "mknapcb4_0",
                       "mknapcb5_0",
                       "mknapcb6_0",
                       "mknapcb7_0",
                       "mknapcb8_0",
                       "mknapcb9_0"]
        
        for file_num in range(len(input_files_name)):
            
            for example_num in range(examples_amount):
                
                # nK = number of knapstacks
                # nI = number of items
                # nK, nI, Capacity, Profits, Weights = Reading_Data.Reading(data_file_name)
                nK, nI, Capacity, Profits, Weights = Reading_Data_mknapcb.Reading(input_files_name[file_num], example_num)
                
                out_file = f'{output_files_name[file_num]}\\{example_num}.txt'
                result = open(out_file, 'a')
                result.write(f"Artificial Bee Colony Algorithm on knapsack\n \n")        
                result.close()

                # 1) writing the results in a text
                # 2) getting the time of algorithm in each iteration
                st = time.time() # get the start time of all     
                
                # getting result by bees :)
                best_bees_of_iterations, best_fitnesses_of_iterations, best_fitnesses_so_far, iteration_num = Bee_Colony_Algorithm(nK, nI, Capacity, Profits, Weights, out_file)

                # clearfiying the bee
                best_final_fitness = max(best_fitnesses_so_far)
                best_final_bee = None
                for b in best_bees_of_iterations:
                    if(b.fitness == best_final_fitness):
                        best_final_bee = b
                        
                # writing the result
                result = open(out_file, 'a')
                # result.write("------------------------ \n")
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
                result.write(f"Number of Employed Bees = {employed_bees_num}\n")
                result.write(f"Number of Onlooker Bees = {onlooker_bees_num}\n")
                result.write(f"Max improvement try = {max_improvement_try}\n")
                result.write(f"cross_over type = {cross_over_type}\n")
                result.write(f"probability of cross-over = {pc}\n")
                result.write(f"probability of mutation = {pm}\n")
                result.write(f"K tournoment percent = {k_tournomet_percent}\n")
                result.write(f"Precedure Type = {percedure_type}\n")
                result.write(f"Number of ABC algorithm's iterations = {iteration_num}\n")
                result.write(f"Limited Time = {cpu_time_limit}")

                # print("---------------------------------")
                # print(f"the best fitness of all: {best_final_fitness}")

                result.close()
                
                iteration_number_list = [i for i in range(1, iteration_num)]
                
                photo = f"{photos_name[file_num]}{example_num}"
                Diagram.diagram(iteration_number_list, best_fitnesses_of_iterations, best_fitnesses_so_far, photo)

if __name__ == '__main__':    
    employed_bees_num = 100  # number of total bees => npop/2 = amount of first population
                        # this must be an even number 
    onlooker_bees_num = 50  # number of iterations in roulette wheel, that select a bee and pass it to improvement-try
    max_improvement_try = 20
    # inner_iteration_of_algorithm = 1000
    pc = 0.7 # the probability of cross-over
    pm = 2 # the probability of mutation (pm/items)
    k_tournomet_percent = 0.1 # in amount of "k_tournomet/items", tournoment will choose, and return the best of them
    # percedure_type = "Tournoment"
    percedure_type = "Roulette Wheel"
    cross_over_type = "one_point"
    p_setZero = 0.05 # the probability of making 1s to 0s in "make_feasible" function
    heuristic_iteration_num = 5; # the number of iterations that heuristic part of algorithm would execute
    
    cpu_time_limit = 300; # second
    
    # file_type = "petersen"
    file_type = "mknapcb"
    
    how_to_read(file_type)
        
    print()