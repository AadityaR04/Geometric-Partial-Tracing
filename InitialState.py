import torch
import numpy as np

class Initial_State():
    '''
    Class to define the input density matrix according to the user's choice
    Some standard density matrices are also defined
    
    Parameters:
        d_level: D level of the system
        No_qudits: number of qudits in the system
        device: device(s) on which the operation is performed
        
    Returns:
        rho: input density matrix
    '''

    def __init__(self, d_level, No_qudits, device):
        self.D = d_level
        self.N = No_qudits
        self.device = device

    def rho_reshape(self, rho):
        '''
        Reshapes the density matrix into a 4D tensor of shape (1, 1, D**N, D**N)
        Also checks if the input density matrix is a valid density matrix (trace = 1)
        
        Parameters:
            rho: input density matrix of shape (D**N, D**N)
            
        Returns:
            rho: reshaped density matrix of shape (1, 1, D**N, D**N)
        '''
        
        if np.round(np.trace(rho.detach().cpu().numpy()), 3) != 1:
            raise Exception('rho is not a valid density matrix')
        
        rho = rho.reshape(1, 1, self.D**self.N, self.D**self.N)
        return rho
    
    def real_custom(self, q):
        '''
        Returns a custom real density matrix according to the user's input
        
        Parameters:
            q: list of tuples containing the position and value of the non-zero elements in the density matrix
            
        Returns:
            rho: reshaped density matrix of shape (1, 1, D**N, D**N)
        '''
        
        psi = torch.zeros((self.D**self.N, 1), dtype= torch.float, requires_grad=False)

        for qudit, value in q:
            psi[qudit] = value

        psi/= torch.norm(psi)
        rho = torch.matmul(psi, psi.t().conj())

        return self.rho_reshape(rho)
    
    def real_random(self):
        '''
        Returns a real Ginibre random density matrix
        
        Parameters:
            None
            
        Returns:
            rho: reshaped density matrix of shape (1, 1, D**N, D**N)
        '''
        
        G = np.random.normal(0, 1, size = (self.D**self.N, self.D**self.N))
        G = torch.tensor(G, dtype= torch.float, requires_grad=False)
        rho = torch.matmul(G, G.t().conj())/torch.trace(torch.matmul(G, G.t().conj()))

        return self.rho_reshape(rho)
    
    def nGHZ(self):
        '''
        Returns a n-qubit GHZ state density matrix
        
        Parameters:
            None
            
        Returns:
            rho: reshaped density matrix of shape (1, 1, D**N, D**N)
        '''
        
        assert self.N >= 3, "Total number of spins should be greater than or equal to 3"
        psi = torch.zeros((self.D**self.N, 1), dtype= torch.float, requires_grad=False)
        # Constructing the GHZ state
        psi[0] = 1
        psi[-1] = 1
        psi/= torch.norm( psi)
        rho = torch.matmul(psi, psi.t().conj())

        return self.rho_reshape(rho)
    
    def nW(self):
        '''
        Returns a n-qubit W state density matrix
        
        Parameters:
            None
            
        Returns:
            rho: reshaped density matrix of shape (1, 1, D**N, D**N)
        '''
        
        assert self.N >= 3, "Total number of spins should be greater than or equal to 3"
        psi = torch.zeros((self.D**self.N, 1), dtype= torch.float, requires_grad=False)
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