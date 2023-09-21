import torch
from utils import Time_Result
# from InitialState import Initial_State
# from QuditPartialTrace import Convolutional_Partial_Trace

#Defining the device
device = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), torch.device("cuda:1" if torch.cuda.is_available() else "cpu")]
# device = torch.device("cpu")

# Define the number of qudits and the dimension of the Hilbert space
Qudit_list = list(range(3, 20))
D_list = [2]

Time_Result(Q_list = Qudit_list, Level_list = D_list, device = device).time_result()

# Q_ = [1, 2, 8, 16]

# rho = Initial_State(d_level= 2, No_qudits= 5, device= device).real_custom(Q_)

# # print(rho)
# # print("------------------")

# Q = [2]

# Partial_Trace = Convolutional_Partial_Trace(input = rho, d_level = 2, qudits = Q, device = device)
# out, time = Partial_Trace.partial_trace()

# print(out)