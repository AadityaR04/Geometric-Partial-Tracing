import os
import json
import datetime
from InitialState import Initial_State
from QuditPartialTrace import Convolutional_Partial_Trace
import torch

class Time_Result():

    def __init__(self, Q_list, Level_list, device):
        self.Qudit_list = Q_list
        self.D_list = Level_list
        self.device = device

    def result(self, label_result, N):

        newpath = r'./Time_Results' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        file_name = "./Time_Results/N_" + str(N) + ".json"

        with open(file_name, 'w') as f:
            json.dump(label_result, f, indent=2)

    def time_result(self):

        print("\nPartial Tracing Time Results")
        print("--------------------------------------------")
        print("Device: ", self.device)
        print("--------------------------------------------")
        print("Time of starting the program: ", datetime.datetime.now())
        print("--------------------------------------------")

        for No_Qudits in self.Qudit_list:
            time_result = []

            print("\nStarting for Qudit No: ", No_Qudits)
            print("--------------------------------------------")
            for D_level in self.D_list:
                
                print("\nD level: ", D_level)
                state = Initial_State(d_level = D_level, No_qudits = No_Qudits, device = self.device)
                rho = state.nW()
                # rho = state.real_random()

                Q, t, label = [], [], []
                
                for q in range(1, No_Qudits):
                    Q.append(q)
                    Partial_Trace = Convolutional_Partial_Trace(input = rho, d_level = D_level, qudits = Q, device = self.device)
                    _, time = Partial_Trace.partial_trace()

                    del _
                    del Partial_Trace
                    torch.cuda.empty_cache()
                    
                    t.append(time)
                    label.append(No_Qudits - q)
                    print("\t No of Qudits traced: ", No_Qudits - q)
                    print("\t Time taken: ", time)
                    print("\n")
                
                time_result.append(t)
                
                del rho
                del state
                torch.cuda.empty_cache()

            print("--------------------------------------------")

            label_result = [time_result, label]

            self.result(label_result, No_Qudits)