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

    file_name = "./Time_Results/N_" + str(N) + ".json"
    with open(file_name, 'r') as f:
        label_result = json.load(f)
    
    time_result = label_result[0]
    
    label = label_result[1]

    plt.figure(figsize=(15, 10))
    

    for index,_ in enumerate(D_list):
        
        mpl.rcParams['axes.linewidth'] = 2
        mpl.rcParams['patch.linewidth'] = 2
        # mpl.rcParams['patch.edgecolor'] = 'black'
        # Change opacity of legend edge
        plt.rcParams['legend.edgecolor'] = 'black'


        plt.plot(label, time_result[index], marker = 'o', label = "D = " + str(D_list[index]), linewidth = 2.0)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.xticks(label)
        plt.legend(fontsize=40)
        plt.xlabel("No of Qubits traced", fontsize=30)
        plt.ylabel("time", fontsize=30)
        plt.title("N = " + str(N), fontsize=40)

    newpath = r'./Plots' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    plt.gcf().set_dpi(300)
    plt.savefig("./Plots/N_" + str(N) + ".png")
    plt.clf()

D_list = [2]
N_list = list(range(3, 16))

for N in N_list:
    plotter(D_list, N)