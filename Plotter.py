from matplotlib import pyplot as plt
import os
import json
import matplotlib as mpl

def plotter(D_list, N):
    '''
    Plots the time taken for partial trace of a density matrix for different D levels and different number of qudits traced out.
    The results are stored in a json file.
    
    Parameters:
        D_list: list of D levels
        N: number of qudits traced out
        
    Returns:
        None
    '''

    # file_name = "./Time_Results/N_" + str(N) + ".json"
    # with open(file_name, 'r') as f:
    #     label_result = json.load(f)
    
    # time_result = label_result[0]
    
    # label = label_result[1]

    plt.figure(figsize=(18, 12))

    Errors = []
    Uncertainty = []

    error_filename = "./Uncertainty/"
    for D in D_list:
        if N <= 4:
            continue
        error_file = error_filename + "D_" + str(D) + "/N_" + str(N) + ".json"
        if os.path.exists(error_file):
            with open(error_file, 'r') as file:
                error_result = json.load(file)
            Uncertainty.append(error_result[0][0])
            Errors.append(error_result[0][1])

    for index, D in enumerate(D_list):
        file_name = "./Time_Results/D_" + str(D) + "/N_" + str(N) + ".json"
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                label_result = json.load(f)
        
        time_result = label_result[0]
        label = label_result[1]

        mpl.rcParams['axes.linewidth'] = 2
        mpl.rcParams['patch.linewidth'] = 2
        # mpl.rcParams['patch.edgecolor'] = 'black'
        # Change opacity of legend edge
        plt.rcParams['legend.edgecolor'] = 'black'

        # plt.plot(label, time_result[index], marker = 'o', label = "D = " + str(D_list[index]), linewidth = 5.0, markersize = 15)
        plt.errorbar(label, time_result[0], yerr=Errors[index], fmt='o-', label="D = " + str(D_list[index]) + r" ($\Delta t$ = " + str(Uncertainty[index]) + "s)", capsize=10, linewidth=1.5, markersize=8, elinewidth = 1.5, markeredgewidth=2)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.xticks(label)
        plt.legend(fontsize=30)
        plt.xlabel("No of Qubits traced", fontsize=30)
        plt.ylabel("time (in sec)", fontsize=30)
        plt.title("N = " + str(N), fontsize=40)

    newpath = r'./Plots' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    plt.gcf().set_dpi(300)
    plt.savefig("./Plots/D_" + str(D_list[-1]) + "_N_" + str(N) + ".png", bbox_inches='tight')
    plt.clf()

D_list = [2, 3, 4, 5, 6]
N_list = list(range(3, 7))

for N in N_list:
    plotter(D_list, N)