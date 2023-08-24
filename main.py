import torch
from InitialState import Initial_State
from QuditPartialTrace import Convolutional_Partial_Trace
import json

# Defining the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Define the number of qudits and the dimension of the Hilbert space
D_level = 2
No_Qudits = 5

state = Initial_State(d_level = D_level, No_qudits = No_Qudits, device = device)

for i in range(10):
    rho = state.real_random()

    Partial_Trace = Convolutional_Partial_Trace(input = rho, d_level = D_level, qudits = [1], device = device)

    reduced_rho, time = Partial_Trace.partial_trace()

    print(reduced_rho)
    print(time)