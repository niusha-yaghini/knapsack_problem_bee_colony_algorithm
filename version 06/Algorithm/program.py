import pandas as pd
import reading_mknapcb as reading_mknapcb
import time
import diagram
from habc import Hybrid_Artificial_Bee_Colony

excel_file_path = 'input_output/input.xlsx'
df = pd.read_excel(excel_file_path)

for index, row in df.iloc[:].iterrows():

    # 0-run_id, 1-file_type, 2-category, 3-problem_num, 4-cpu_time_limit, 5-employed_bees_num, 6-max_try_improve, 
    # 7-pc, 8-pm(pm/items), 9-selection_type, 10-k_tournomet, 11-crossover_type

    row_values = list(row.iloc)

    run_id, category, problem_num, cpu_time_limit, employed_bees_num, onlooker_bees_num, max_try_improve, crossover_type, pc_onePoint, pc_uniForm, crossover_neighbor_selection, cn_k_tournomet, pm, make_feasible_check, onlooker_selection, i_k_tournomet, p_scout_delete, heuristic_phase_check, heuristic_iteration_num, heuristic_selection, h_k_tournoment, p_set_zero = row_values[0], row_values[1], row_values[2], row_values[3], row_values[4], row_values[5], row_values[6], row_values[7], row_values[8], row_values[9], row_values[10], row_values[11], row_values[12], row_values[13], row_values[14], row_values[15], row_values[16], row_values[17], row_values[18], row_values[19], row_values[20], row_values[21]

    # nK = number of knapstacks
    # nI = number of items
    nK, nI, Capacity, Profits, Weights = reading_mknapcb.reading(category, problem_num)
    
    result_file_name = f'input_output/{run_id}.txt'
    result = open(result_file_name, 'a')
    result.write(f"Modified Hybrid Artificial Bee Colony Algorithm on knapsack\n \n")        
    result.close()

    # 1) writing the results in a text
    # 2) getting the time of algorithm in each iteration
    st = time.time() # get the start time of all     
        
    # creating an abc object
    abc = Hybrid_Artificial_Bee_Colony(index, run_id, category, problem_num, nK, nI, Capacity, Profits, Weights, cpu_time_limit, employed_bees_num, onlooker_bees_num, max_try_improve, crossover_type, pc_onePoint, pc_uniForm, crossover_neighbor_selection, cn_k_tournomet, pm, make_feasible_check, onlooker_selection, i_k_tournomet, p_scout_delete, heuristic_phase_check, heuristic_iteration_num, heuristic_selection, h_k_tournoment, p_set_zero, result_file_name)

    # getting result
    best_fitnesses_of_iterations, best_fitnesses_so_far, best_final_bee, best_final_fitness, total_iteration_num, best_fitness_iteration_num, best_fitness_time, time_number_list = abc.optimize()

    # writing the result in txt
    abc.write_results(best_fitnesses_of_iterations, best_final_bee, best_final_fitness, st)

    # writing in excel 
    abc.write_excel(run_id, category, problem_num, cpu_time_limit, employed_bees_num, onlooker_bees_num, max_try_improve, crossover_type, pc_onePoint, pc_uniForm, crossover_neighbor_selection, cn_k_tournomet, pm, make_feasible_check,onlooker_selection, i_k_tournomet, p_scout_delete, heuristic_phase_check, heuristic_iteration_num, heuristic_selection,h_k_tournoment, p_set_zero, best_final_fitness, best_fitness_iteration_num, best_fitness_time, total_iteration_num)
        
    photo = f"input_output/{run_id}"
    diagram.diagram(time_number_list, best_fitnesses_of_iterations, best_fitnesses_so_far, photo)

# End of program ------------------------
