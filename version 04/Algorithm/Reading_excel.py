import pandas as pd
import Reading_Data_mknapcb
import time
from datetime import datetime
import Artificial_Bee_Colony
import copy
import Diagram
import Bees


def Bee_Colony_Algorithm(run_id, cpu_time_limit, employed_bees_num, onlooker_bees_num,
                         max_improvement_try, pc, pm, percedure_type, k_tournomet_percent, cross_over_type,
                         p_setZero, heuristic_iteration_num,
                         nK, nI, Capacity, Profits, Weights, result_file_name):
    
    population = []
        
    best_fitnesses_of_iteration = []
    best_fitnesses_so_far = []
    best_fitness_so_far = 0
    best_bee_so_far = Bees.Bee
    
    start_limitation_time = time.time()
    elapsed_limitation_time = 0
    
    iteration_num = 1
    best_fitness_iteration_num = iteration_num

    currentTime = datetime.now().strftime("%H:%M:%S")
    print(f"run_id -> {run_id}: {currentTime}")
    
    start_t = time.time()

    while(elapsed_limitation_time<cpu_time_limit):

        result = open(result_file_name, 'a')    
        iteration_num+=1
 
        ABC = Artificial_Bee_Colony.ABC_algorithm(employed_bees_num, nK, nI, Capacity, Profits, Weights, 
                onlooker_bees_num, max_improvement_try, pc, pm, k_tournomet_percent, percedure_type,
                cross_over_type, p_setZero, heuristic_iteration_num)
        
        ABC.employed_bees(population)
        ABC.onlooker_bees(population)
        best_bee_of_iteration, best_fitness_of_iteration = ABC.finding_best_bee(population)
        
        best_fitnesses_of_iteration.append(copy.deepcopy(best_fitness_of_iteration))
        
        if(best_fitness_of_iteration > best_fitness_so_far):
            best_fitness_so_far = best_fitness_of_iteration
            best_bee_so_far = best_bee_of_iteration
            print(f"best fitness so far: {best_fitness_so_far}")
            best_fitness_iteration_num = iteration_num
            end_t = time.time()   
                   
        best_fitnesses_so_far.append(best_fitness_so_far)            

        ABC.scout_bees(population)
        
        current_limitation_time = time.time()
        elapsed_limitation_time = current_limitation_time-start_limitation_time;
                        
        result.close()    
    
    best_fitness_time = end_t - start_t
    return best_fitnesses_of_iteration, best_fitnesses_so_far, best_bee_so_far, best_fitness_so_far, iteration_num, best_fitness_iteration_num, best_fitness_time


def write_excel(run_id, best_final_fitness, best_fitness_iteration_num, best_fitness_time, total_iteration_num, cpu_time_limit):
    file_path = 'input_output/output.xlsx'

    # Create a DataFrame with the three items
    # new_data = {'run_id': run_id,
    #             'best_final_fitness': best_final_fitness,
    #             'best_fitness_iteration_num': best_fitness_iteration_num,
    #             'best_fitness_time': best_fitness_time,
    #             'total_iteration_num': total_iteration_num,
    #             'cpu_time_limit' : cpu_time_limit}
    
    df = pd.read_excel(file_path)
    new_data = {'run_id': run_id,
                'best_final_fitness': best_final_fitness,
                'best_fitness_iteration_num': best_fitness_iteration_num,
                'best_fitness_time': best_fitness_time,
                'total_iteration_num': total_iteration_num,
                'cpu_time_limit' : cpu_time_limit}

    # Update the specific row with the new data
    # df.iloc[run_id, df.columns.get_loc('run_id')] = new_data['run_id']
    # df.iloc[run_id, df.columns.get_loc('best_final_fitness')] = new_data['best_final_fitness']
    # df.iloc[run_id, df.columns.get_loc('best_fitness_iteration_num')] = new_data['best_fitness_iteration_num']
    # df.iloc[run_id, df.columns.get_loc('best_fitness_time')] = new_data['best_fitness_time']
    # df.iloc[run_id, df.columns.get_loc('total_iteration_num')] = new_data['total_iteration_num']
    # df.iloc[run_id, df.columns.get_loc('cpu_time_limit')] = new_data['cpu_time_limit']

    df = df.append(new_data, ignore_index=True)

    # df.loc[df['run_id'] == run_id, new_data.keys()] = new_data.values()

    df.to_excel(file_path, index=False)


def run_example(run_id, category, problem_num, cpu_time_limit, employed_bees_num, onlooker_bees_num,
                max_improvement_try, pc, pm, percedure_type, k_tournomet_percent, cross_over_type,
                p_setZero, heuristic_iteration_num):

    # nK = number of knapstacks
    # nI = number of items
    nK, nI, Capacity, Profits, Weights = Reading_Data_mknapcb.Reading(category, problem_num)
    
    out_file = f'input_output/{run_id}.txt'
    result = open(out_file, 'a')
    result.write(f"Artificial Bee Colony Algorithm on knapsack\n \n")        
    result.close()

    # 1) writing the results in a text
    # 2) getting the time of algorithm in each iteration
    st = time.time() # get the start time of all     
    
    # getting result by bees :)
    best_fitnesses_of_iterations, best_fitnesses_so_far, best_final_bee, best_final_fitness, total_iteration_num, best_fitness_iteration_num, best_fitness_time = Bee_Colony_Algorithm(
        run_id, cpu_time_limit, employed_bees_num, onlooker_bees_num, max_improvement_try, pc, pm,
        percedure_type, k_tournomet_percent, cross_over_type, p_setZero, heuristic_iteration_num,
        nK, nI, Capacity, Profits, Weights, out_file)
            
    # writing the result
    result = open(out_file, 'a')
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
    result.write(f"Number of ABC algorithm's iterations = {total_iteration_num}\n")
    result.write(f"Limited Time = {cpu_time_limit}")

    result.close()
    
    # writing in excel 
    write_excel(run_id, best_final_fitness, best_fitness_iteration_num, best_fitness_time, total_iteration_num, cpu_time_limit)
    
    iteration_number_list = [i for i in range(1, total_iteration_num)]
    
    photo = f"input_output/{run_id}"
    Diagram.diagram(iteration_number_list, best_fitnesses_of_iterations, best_fitnesses_so_far, photo)


def reading_excel(excel_file_path):

    excel_file_path = 'input_output/input.xlsx'

    df = pd.read_excel(excel_file_path)

    for index, row in df.iloc[:].iterrows():

        # 0-run_id, 1-file_type, 2-category, 3-problem_num, 4-cpu_time_limit, 5-employed_bees_num, 6-onlooker_bees_num,
        # 7-max_improvement_try, 8-pc, 9-pm(pm/items), 10-percedure_type, 11-k_tournomet_percent, 12-cross_over_type,
        # 13-p_set_zero_prob, 14-heuristic_iteration_num

        row_values = list(row.iloc)
        run_example(row_values[0], row_values[2], row_values[3], row_values[4], row_values[5],
                    row_values[6], row_values[7], row_values[8], row_values[9], row_values[10],
                    row_values[11], row_values[12], row_values[13], row_values[14])
        