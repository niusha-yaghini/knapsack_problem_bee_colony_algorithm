import numpy as np

def Reading(file_name):
    data = open(f"{file_name}", "r")
    
    info = data.readline().split(" ")
    nI = int(info[1])
    nK = int(info[2])
    
    Profits = []
    for p in range(72):
        pro = [float(x) for x in data.readline().split(" ") if x.strip()]
        Profits.extend(pro)
        
    Weights = []
    for knapsack in range(nK):
        Weight = []
        for w in range(72):
            wei = [float(x) for x in data.readline().split(" ") if x.strip()]
            Weight.extend(wei)
        Weights.append(Weight)
                
    Capacity = []
    for c in range(71):
        cap = [float(x) for x in data.readline().split(" ") if x.strip()]
        Capacity.extend(cap)

        
    return nK, nI, Capacity, Profits, Weights