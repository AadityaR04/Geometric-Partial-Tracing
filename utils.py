import os
import json
from InitialState import Initial_State
from QuditPartialTrace import Convolutional_Partial_Trace

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

        for No_Qudits in self.Qudit_list:
            time_result = []
            for D_level in self.D_list:

                state = Initial_State(d_level = D_level, No_qudits = No_Qudits, device = self.device)
                rho = state.real_random()

                Q, t, label = [], [], []
                
                for q in range(1, No_Qudits):
                    Q.append(q)
                    Partial_Trace = Convolutional_Partial_Trace(input = rho, d_level = D_level, qudits = Q, device = self.device)
                    reduced_rho, time = Partial_Trace.partial_trace()
                    t.append(time)
                    label.append(No_Qudits - q)
                
                time_result.append(t)
            
            label_result = [time_result, label]

            self.result(label_result, No_Qudits)