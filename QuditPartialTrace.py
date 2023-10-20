import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Block_Split_Loader(Dataset):
    def __init__(self, reduced_block):
        self.len = reduced_block.shape[0]
        self.reduced_block = reduced_block
        
    def __getitem__(self, index):
        return self.reduced_block[index]
    
    def __len__(self):
        return self.len


class Convolutional_Partial_Trace(nn.Module):
    
    def __init__(self, input, d_level, qudits, device):
        super(Convolutional_Partial_Trace, self).__init__()
        self.rho = input
        self.D = d_level
        self.Q = qudits
        self.M = len(qudits)
        self.dim_D = self.D**2
        self.device = device
        
        del input
        
        self.log_n_x(self.rho, self.D)
        self.block_splitting()
        self.kernel_matrix()
        
        self.pt = nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=self.dimension, stride = self.dimension, padding = 0, bias=False)
        with torch.no_grad():
            self.pt.weight = nn.Parameter(self.kernel)
            
    def log_n_x(self, x, base):
        self.N = int(torch.log(torch.tensor(x.shape[2]))/torch.log(torch.tensor(base)))

    def block_splitting(self):
        temp = self.rho
        self.channels = 1 # Number of channels in the kernel

        # If M = 0, then the partial trace is just the trace
        reduced_block = temp
        self.block_dim = self.D ** (self.N + 1)
        
        for i in range(self.M):
            if i == 0:
                q_i_0 = 0
            else:
                q_i_0 = self.Q[i-1]
            
            q_i_1 = self.Q[i]

            self.block_dim = self.D ** (self.N - q_i_1 + 1)
            No_of_blocks = self.D ** (q_i_1 - q_i_0 - 1)
            self.channels *= No_of_blocks

            # For j = 0
            reduced_block = temp[:,:,0*self.block_dim:(0+1)*self.block_dim, 0*self.block_dim:(0+1)*self.block_dim]

            for j in range(1, No_of_blocks):
                # Divide into blocks
                block = temp[:,:,j*self.block_dim:(j+1)*self.block_dim, j*self.block_dim:(j+1)*self.block_dim]
                # Concatenate the blocks into channels
                reduced_block = torch.cat((reduced_block, block), dim = 1)

            if i != self.M-1:
                # Split into groups of dim
                column_split = torch.chunk(reduced_block, self.D, dim = 3)
                column = torch.stack(column_split, dim = 0).reshape(-1, self.channels, self.block_dim, self.block_dim//self.D)
                row_split = torch.chunk(column, self.D, dim = 2)
                reduced_block = torch.stack(row_split, dim = 0).reshape(-1, self.channels, self.block_dim//self.D, self.block_dim//self.D)
                
                temp = reduced_block
                del column_split, column, row_split
            
        del temp
        torch.cuda.empty_cache()
        
        block_loader = DataLoader(dataset= Block_Split_Loader(reduced_block), batch_size = reduced_block.shape[0], shuffle = False) 
        del reduced_block
        
        return block_loader
        

    def kernel_matrix(self):
        self.dimension = self.block_dim//self.D
        self.kernel = torch.eye(self.dimension, dtype = torch.float).repeat(1, self.channels, 1, 1)

    
    def forward(self, input):
        
        del self.rho
        torch.cuda.empty_cache()

        output = self.pt(input)
        # print("\tIn Model: input size", input.size(), "output size", output.size())
        return output

    

class Matrix_Maker(Convolutional_Partial_Trace):
    
    def matrix_reassembly(self, reduced_tensor):
        tensor = reduced_tensor.squeeze()
        del reduced_tensor
        torch.cuda.empty_cache()

        for i in range(1, self.M):
            pow = i
            block = torch.chunk(tensor, self.dim_D, dim = 0)
            tensor = torch.stack(block, dim = 1).reshape(-1, self.D, self.D, self.D**(pow), self.D**(pow)).transpose(3, 2).reshape(-1, self.D**(pow+1), self.D**(pow+1))
            
            del block
            torch.cuda.empty_cache()

        return tensor.squeeze()