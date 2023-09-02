import torch
from utils import Time_Result
from Plotter import Plotter

#Defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of qudits and the dimension of the Hilbert space
Qudit_list = list(range(2, 19))
D_list = list(range(2, 5))

Time_Result(Q_list = Qudit_list, Level_list = D_list, device = device).time_result()
