import torch
import numpy as np

class Initial_State():

    def __init__(self, d_level, No_qudits, device):
        self.D = d_level
        self.N = No_qudits
        self.device = device

    def rho_reshape(self, rho):
        
        test = rho.detach().cpu().numpy()
        if np.round(np.trace(test), 3) != 1:
            raise Exception('rho is not a valid density matrix')
        
        rho = rho.reshape(1, 1, self.D**self.N, self.D**self.N)
        return rho
    
    def real_custom(self, Q_):
        psi = torch.zeros((self.D**self.N, 1), dtype= torch.float)

        for qudit, value in Q_:
            psi[qudit] = value

        psi/= torch.norm(psi)
        rho = torch.matmul(psi, psi.t().conj())

        return self.rho_reshape(rho)
    
    def real_random(self):
        G = np.random.normal(0, 1, size = (self.D**self.N, self.D**self.N))
        G = torch.tensor(G, dtype= torch.float)
        rho = torch.matmul(G, G.t().conj())/torch.trace(torch.matmul(G, G.t().conj()))

        return self.rho_reshape(rho)
    
    def nGHZ(self):
        assert self.N >= 3, "Total number of spins should be greater than or equal to 3"
        psi = torch.zeros((self.D**self.N, 1), dtype= torch.float)
        # Constructing the GHZ state
        psi[0] = 1
        psi[-1] = 1
        psi/= torch.norm( psi)
        rho = torch.matmul(psi, psi.t().conj())

        return self.rho_reshape(rho)
    
    def nW(self):
        assert self.N >= 3, "Total number of spins should be greater than or equal to 3"
        psi = torch.zeros((self.D**self.N, 1), dtype= torch.float)
        # Constructing the W state
        psi[0] = 1
        for i in range(1, self.N):
            psi[self.D**i] = 1
        psi/= torch.norm(psi)
        rho = torch.matmul(psi, psi.t().conj())

        return self.rho_reshape(rho)

    
    # def complex_random(self):
    #     G = np.zeros((self.D**self.N, self.D**self.N), dtype = np.complex_)
    #     for i in range(self.D**self.N):
    #         for j in range(self.D**self.N):
    #             G[i][j] = complex(np.random.normal(0, 1, size= None), np.random.normal(0, 1, size= None))

    #         G = torch.tensor(G, device = self.device)
    #         rho = torch.matmul(G, G.t().conj())/torch.trace(torch.matmul(G, G.t().conj()))

    #     return self.rho_reshape(rho)
    
    # def complex_custom(self, psi):

    #     psi = torch.tensor(psi, device = self.device)
    #     psi/= torch.norm(psi)
    #     rho = torch.matmul(psi, psi.t().conj())

    #     return self.rho_reshape(rho)