import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Block_Split_Loader(Dataset):
    '''
    Loads the block splitted tensor into a DataLoader object
    
    Parameters:
        reduced_block: block splitted tensor
        
    Returns:
        None
    '''
    
    def __init__(self, reduced_block):
        self.len = reduced_block.shape[0]
        self.reduced_block = reduced_block
        
    def __getitem__(self, index):
        return self.reduced_block[index]
    
    def __len__(self):
        return self.len


class Convolutional_Partial_Trace(nn.Module):
    '''
    Class to perform the partial trace operation on a density matrix using 3D convolution
    
    Parameters:
        input: input density matrix
        d_level: D level of the system
        qudits: list of qudits to be traced out
        device: device(s) on which the operation is performed
    '''
    
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
        
        '''
        The 3D convolution is performed using the nn.Conv2d module of PyTorch. The gradients are turned off as it is a fixed kernel.
        '''
        
        self.pt = nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=self.dimension, stride = self.dimension, padding = 0, bias=False)
        with torch.no_grad():
            self.pt.weight = nn.Parameter(self.kernel, requires_grad=False)
            
    def log_n_x(self, x, base):
        '''
        Returns the log of x in base
        
        Parameters:
            x: input tensor
            base: base of the log
            
        Returns:
            log of x in base
        '''
        
        self.N = int(torch.log(torch.tensor(x.shape[2]))/torch.log(torch.tensor(base)))

    def block_splitting(self):
        '''
        Splits the tensor into blocks arranged in channels and batches according to the number of qudits traced out and loads it into a DataLoader object
        
        Parameters:
            None
            
        Returns:
            None        
        '''
        
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
        '''
        Creates the kernel matrix for convolution
        
        Parameters:
            None
            
        Returns:
            None
        '''
        
        self.dimension = self.block_dim//self.D
        self.kernel = torch.eye(self.dimension, dtype = torch.float).repeat(1, self.channels, 1, 1)
    
    def forward(self, input):
        '''
        Performs the 3D convolution on the block splitted tensor
        
        Parameters:
            input: block splitted tensor
            
        Returns:
            output: output of the convolution which is the block splitted resultant tensor obtained after partial trace
        '''
        
        del self.rho
        torch.cuda.empty_cache()

        output = self.pt(input)
        # print("\tIn Model: input size", input.size(), "output size", output.size())
        return output


def trace(loader, output, device, Partial_Trace):
    '''
    Performs the partial trace operation on the input tensor
    
    Parameters:
        loader: DataLoader object containing the block splitted tensor
        output: list to store the output of the partial trace operation
        device: device(s) on which the operation is performed
        Partial_Trace: Partial_Trace object
        
    Returns:
        reduced_tensor: output of the partial trace operation in block splitted form
    '''
    
    for batch in loader:
        batch = batch.to(device)
        out = Partial_Trace(batch)
        output.append(out)
    
    reduced_tensor = torch.cat(output, dim = 0)
    
    return reduced_tensor


def Matrix_Maker(input, D, Q):
    '''
    Reassembles the block splitted tensor into a density matrix which is the final output of the partial trace operation
    
    Parameters:
        input: block splitted tensor
        D: D level of the system
        Q: list of qudits to be traced out
        
    Returns:
        tensor: final density tensor after partial trace
    '''
    
    tensor = input.squeeze()
    del input
    torch.cuda.empty_cache()
    
    M = len(Q)
    
    for i in range(1, M):
        pow = i
        block = torch.chunk(tensor, D**2, dim = 0)
        tensor = torch.stack(block, dim = 1).reshape(-1, D, D, D**(pow), D**(pow)).transpose(3, 2).reshape(-1, D**(pow+1), D**(pow+1))
        
        del block
        torch.cuda.empty_cache()
        
    return tensor.squeeze()