import sys
sys.dont_write_bytecode = True

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
from scipy.optimize import curve_fit

class Time_Result():
    '''
    Class to compute the time taken for partial trace of a density matrix for different D-level, N particle systems.
    The results are stored in a json file.
    Parameters:
        Q_list: list of number of qudits in the system
        Level_list: list of D levels
        device: device(s) on which the operation is performed

    Returns:
        None
    '''

    def __init__(self, Q_list, Level_list, device):
        self.Qudit_list = Q_list
        self.D_list = Level_list
        self.device = device

    def result(self, label_result, N, D):
        '''
        Saves the time results in a json file.
        Parameters:
            label_result: list containing time results and labels
            N: number of qudits traced out
            D: D level of the system

        Returns:
            None
        '''

        newpath = r'./Time_Results/D_' + str(D) + '/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        file_name = newpath + "N_" + str(N) + ".json"

        with open(file_name, 'w') as f:
            json.dump(label_result, f, indent=2)

    def time_result(self):
        '''
        Computes the time taken for partial trace of a density matrix for different D levels and different number of qudits traced out.
        The results are stored in a json file.
        Parameters:
            None

        Returns:
            None
        '''

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

                label_result = [time_result, label]
                self.result(label_result, No_Qudits, D_level)
            print("--------------------------------------------")

            

    def exp_decay(self, x, A, k, C):
        '''
        Exponential decay function for curve fitting.
        Parameters:
            x: independent variable (number of qubits traced)
            A: amplitude
            k: decay constant
            C: offset

        Returns:
            None
        '''
        return A * np.exp(-k * x) + C
    
    def uncertainty(self):
        '''
        Computes the uncertainty in the time taken for partial trace of a density matrix for different D levels and different number of qudits traced out.
        The results are stored in a json file.
        Parameters:
            None

        Returns:
            None
        '''
        for No_Qudits in self.Qudit_list:
            if No_Qudits <= 4:
                continue

            Uncertainty = []

            for D_level in self.D_list:
                filename = "./Time_Results/D_" + str(D_level) + "/N_" + str(No_Qudits) + ".json"
                with open(filename, 'r') as f:
                    data = json.load(f)
                time_result = np.array(data[0][0])
                label = np.array(data[1])

                popt, _ = curve_fit(self.exp_decay, label, time_result, p0=(1, 0.1, 0), maxfev=10000, method='trf')
                error = np.abs(time_result - self.exp_decay(label, *popt))
                std_dev = np.sqrt(np.sum(error**2) / (len(error.tolist()) - len(popt)))

                Uncertainty.append([np.round(std_dev, 4), error.tolist()])
            error_filename = "./Uncertainty/D_" + str(D_level) + "/"
            if not os.path.exists(error_filename):
                os.makedirs(error_filename)
            error_file = error_filename + "N_" + str(No_Qudits) + ".json"
            with open(error_file, 'w') as f:
                json.dump(Uncertainty, f)

class Output_Result():
    '''
    Class to compute the output of the partial trace operation.
    This class saves the output density matrix in a text file and prints the result.

    Parameters:
        input: input density matrix
        D_level: D level of the system
        Qudits: list of qudits to be traced out
        device: device(s) on which the operation is performed

    Returns:
        None
    '''
    
    def __init__(self, input, D_level, Qudits, device):
        self.rho = input
        self.D = D_level
        self.Q = Qudits
        self.device = device
        
        del input
    
    def result(self, output):
        '''
        Saves the output density matrix to a text file.

        Parameters:
            output: output density matrix

        Returns:
            None
        '''

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
        
        del output
        torch.cuda.empty_cache()
    
    
    def print_result(self, output):
        '''
        Prints the output density matrix and other information.
        
        Parameters:
            output: output density matrix

        Returns:
            None
        '''
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
        '''
        Computes the output of the partial trace operation.

        Parameters:
            None

        Returns:
            None
        '''
        Partial_Trace = Convolutional_Partial_Trace(input = self.rho, d_level = self.D, qudits = self.Q, device = self.device)
        
        del self.rho
        torch.cuda.empty_cache()
        
        loader = Partial_Trace.block_splitting()
        
        if torch.cuda.device_count() > 1:
            Partial_Trace = nn.DataParallel(Partial_Trace)
        Partial_Trace.to(self.device)
        
        output = []
        reduced_tensor = QPT.trace(loader, output, self.device, Partial_Trace)
        
        del loader, output, Partial_Trace
        torch.cuda.empty_cache()
        
        final_result = QPT.Matrix_Maker(input = reduced_tensor, D = self.D, Q = self.Q)
        
        del reduced_tensor
        torch.cuda.empty_cache()
        
        self.result(final_result)
        self.print_result(final_result)