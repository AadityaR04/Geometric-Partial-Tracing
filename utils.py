import os
import json
import time
import datetime
from InitialState import Initial_State
import QuditPartialTrace as QPT
from QuditPartialTrace import Convolutional_Partial_Trace
import torch
import torch.nn as nn
import numpy as np

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
        print("No of Devices: ", torch.cuda.device_count())
        print("--------------------------------------------")
        
        if torch.cuda.device_count() > 1:
            print("Devices: ")
            for i in range(torch.cuda.device_count()):
                print("Device No " + str(i) + ":", end = " ")
                print(torch.cuda.get_device_name(i), end = " ")
                print("\t", end = " ")
            print("\n")
        else:
            print("Device No 0: ", torch.cuda.get_device_name(0))
            
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
                    output = []
                    Q.append(q)
                    
                    Partial_Trace = Convolutional_Partial_Trace(input = rho, d_level = D_level, qudits = Q, device = self.device)
                    
                    t1 = time.time()
                    
                    loader = Partial_Trace.block_splitting()
                    
                    if torch.cuda.device_count() > 1:
                        Partial_Trace = nn.DataParallel(Partial_Trace)
                    Partial_Trace.to(self.device)
                    
                    reduced_tensor = QPT.trace(loader, output, self.device, Partial_Trace)
                    _ = QPT.Matrix_Maker(input = reduced_tensor, D = D_level, Q = Q)
                    t2 = time.time()
                    
                    Time = t2 - t1

                    del output
                    del Partial_Trace
                    del _
                    del reduced_tensor
                    del loader
                    
                    torch.cuda.empty_cache()
                    
                    t.append(Time)
                    label.append(No_Qudits - q)
                    print("\t No of Qudits traced: ", No_Qudits - q)
                    print("\t Time taken: ", Time)
                    print("\n")
                
                time_result.append(t)
                
                del rho
                del state
                torch.cuda.empty_cache()

            print("--------------------------------------------")

            label_result = [time_result, label]

            self.result(label_result, No_Qudits)
            
class Output_Result():
    
    def __init__(self, input, D_level, Qudits, device):
        self.rho = input
        self.D = D_level
        self.Q = Qudits
        self.device = device
        
        del input
    
    def result(self, output):

        newpath = r'./PartialTrace_Results' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        output = output.detach().cpu().numpy()
        output = output.squeeze()
        output = np.round(output, 3)
        
        label = "rho_"
        
        for q in self.Q:
            label += str(q)
        
        label += "_D_" + str(self.D) + ".txt"
        
        np.savetxt("./PartialTrace_Results/" + label, output, fmt = '%1.3f')        
    
    
    def print_result(self, output):
        print("\nPartial Tracing Results")
        print("--------------------------------------------")
        print("No of Devices: ", torch.cuda.device_count())
        print("--------------------------------------------")
        
        if torch.cuda.device_count() > 1:
            print("Devices: ")
            for i in range(torch.cuda.device_count()):
                print("Device No " + str(i) + ":", end = " ")
                print(torch.cuda.get_device_name(i), end = " ")
                print("\t", end = " ")
            print("\n")
        else:
            print("Device No 0: ", torch.cuda.get_device_name(0))
            
        print("--------------------------------------------")
        print("Time of starting the program: ", datetime.datetime.now())
        print("--------------------------------------------")
        
        print("Output Matrix = rho_", end = "")
        for q in self.Q:
            print(str(q), end = "")
        
        print("\tFor D level = ", self.D)
        print("--------------------------------------------")
        print(output.detach().cpu().numpy())
        print("\n")
        

    def output(self):
        Partial_Trace = Convolutional_Partial_Trace(input = self.rho, d_level = self.D, qudits = self.Q, device = self.device)
        
        loader = Partial_Trace.block_splitting()
        
        if torch.cuda.device_count() > 1:
            Partial_Trace = nn.DataParallel(Partial_Trace)
        Partial_Trace.to(self.device)
        
        output = []
        reduced_tensor = QPT.trace(loader, output, self.device, Partial_Trace)
        final_result = QPT.Matrix_Maker(input = reduced_tensor, D = self.D, Q = self.Q)
        
        self.result(final_result)
        self.print_result(final_result)