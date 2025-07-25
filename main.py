import sys
sys.dont_write_bytecode = True

import torch
from utils import Time_Result, Output_Result
from InitialState import Initial_State

# #Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
To calculate the time taken for partial trace of a density matrix, we need to define the following:
    1. The D level of the system (D_list)
    2. The number of qudits in the system (Qudit_list)

The following code snippet will calculate the time taken for partial trace of a density matrix for different number of qudits and different D levels along with the uncertainty in the time taken.
'''

# Define the number of qudits and the dimension of the Hilbert space
D_list = [2, 3, 4, 5, 6]
Qudit_list = list(range(3, 8))

TR = Time_Result(Q_list = Qudit_list, Level_list = D_list, device = device)
TR.time_result()
TR.uncertainty()

'''
To calculate the partial trace of a density matrix, we need to define the following:
    1. The D level of the system (D)
    2. The qudits to be traced out (Q)

The input density matrix is defined in InitialState.py
For custom density matrices, q is the position and value of the non-zero elements in the density matrix.

The following code snippet will calculate the partial trace of a density matrix for a given D level and qudits to be traced out.
'''

# D = 2
# q = [(1, 1), (2, 1), (8, 1), (16, 1)]

# rho = Initial_State(d_level= 2, No_qudits= 5, device= device).real_custom(q)

# # print(rho)
# # print("------------------")

# Q = [2]

# Traced_Result = Output_Result(input= rho, D_level = D, Qudits = Q, device = device)
# Traced_Result.output()