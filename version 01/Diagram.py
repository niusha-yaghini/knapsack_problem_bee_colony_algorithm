import matplotlib.pyplot as plt

def diagram(iteration_numbers, best_fitnesses_each_iter, best_fitnesses_so_far, photo_name):

    fig, ax = plt.subplots()
    each_iter,  = plt.plot(iteration_numbers, best_fitnesses_each_iter, label='best fitness of each iteration')
    so_far, = plt.plot(iteration_numbers, best_fitnesses_so_far, label='best fitness so far')
    
    ax.set_title(f"Artificial Bee Colony Algorithm")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fitness")
    ax.legend(handles=[each_iter, so_far])

    name = f"{photo_name}" + '.png'

    plt.savefig(name)
    plt.show()