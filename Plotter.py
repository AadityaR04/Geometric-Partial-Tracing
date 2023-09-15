from matplotlib import pyplot as plt
import os
import json

class Plotter():

    def plotter(self, D_list, N):

        file_name = "./Time_Results/N_" + str(N) + ".json"
        with open(file_name, 'r') as f:
            label_result = json.load(f)
        
        time_result = label_result[0]
        
        label = label_result[1]
        

        for index,_ in enumerate(D_list):
            
            plt.plot(label, time_result[index], marker = 'o')
            plt.xticks(label)
            plt.legend(D_list)
            plt.xlabel("No of Qubits traced")
            plt.ylabel("time")
            plt.title("N = " + str(N))

        newpath = r'./Plots' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        plt.savefig("./Plots/N_" + str(N) + ".png")
        plt.clf()

D_list = [2]
N_list = list(range(2, 20))

for N in N_list:
    Plotter().plotter(D_list, N)