from matplotlib import pyplot as plt
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
            # plt.scatter(label, time_result[index])
            plt.xlabel("No of Qubits traced")
            plt.ylabel("time")
            plt.title("N = " + str(N))
        
        plt.show()

D_list = list(range(2, 5))
Plotter().plotter(D_list, 5)
