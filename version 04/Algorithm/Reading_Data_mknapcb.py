import math

def Reading(category, problem_num):
    file_name = f'Algorithm/mknapcb/{category}.txt'
    
    data = open(file_name, "r")
    data.readline()
    
    info = data.readline().split(" ")
    nI = int(info[1])
    nK = int(info[2])
    
    items_amount = math.ceil(nI/7)
    knapsacks_amount = math.ceil(nK/7)
    
    line_to_pass_amount = problem_num * (((1+nK) * math.ceil(nI/7)) + math.ceil(nK/7) + 1)
    for i in range(line_to_pass_amount):
        data.readline()
    
    Profits = []
    for p in range(items_amount):
        pro = [float(x) for x in data.readline().split(" ") if x.strip()]
        Profits.extend(pro)
        
    Weights = []
    for knapsack in range(nK):
        Weight = []
        for w in range(items_amount):
            wei = [float(x) for x in data.readline().split(" ") if x.strip()]
            Weight.extend(wei)
        Weights.append(Weight)
                
    Capacity = []
    for c in range(knapsacks_amount):
        cap = [float(x) for x in data.readline().split(" ") if x.strip()]
        Capacity.extend(cap)

        
    return nK, nI, Capacity, Profits, Weights