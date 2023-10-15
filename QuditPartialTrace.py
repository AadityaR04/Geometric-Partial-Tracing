import torch
import torch.nn.functional as F
import torch.distributed as dist
import time

class Convolutional_Partial_Trace:
    
    def __init__(self, input, d_level, qudits, device):
        self.rho = input
        self.D = d_level
        self.Q = qudits
        self.M = len(qudits)
        self.dim_D = self.D**2
        self.device = device

        del input

    def log_n_x(self, x, base):
        self.N = int(torch.log(torch.tensor(x.shape[2], device = self.device[1]))/torch.log(torch.tensor(base, device = self.device[1])))

    def block_splitting(self):
        temp = self.rho.to(self.device[1])
        del self.rho
        channels = 1 # Number of channels in the kernel

        # If M = 0, then the partial trace is just the trace
        reduced_block = temp.to(self.device[0])
        block_dim = self.D ** (self.N + 1)

        torch.cuda.empty_cache()
        
        for i in range(self.M):
            if i == 0:
                q_i_0 = 0
            else:
                q_i_0 = self.Q[i-1]
            
            q_i_1 = self.Q[i]

            block_dim = self.D ** (self.N - q_i_1 + 1)
            No_of_blocks = self.D ** (q_i_1 - q_i_0 - 1)
            channels *= No_of_blocks

            # For j = 0
            reduced_block = temp[:,:,0*block_dim:(0+1)*block_dim, 0*block_dim:(0+1)*block_dim].to(self.device[0])

            for j in range(1, No_of_blocks):
                # Divide into blocks
                block = temp[:,:,j*block_dim:(j+1)*block_dim, j*block_dim:(j+1)*block_dim].to(self.device[1])
                # Concatenate the blocks into channels
                reduced_block = torch.cat((reduced_block, block), dim = 1).to(self.device[0])

            if i != self.M-1:
                # Split into groups of dim
                column_split = torch.chunk(reduced_block, self.D, dim = 3)
                column = torch.stack(column_split, dim = 0).reshape(-1, channels, block_dim, block_dim//self.D).to(self.device[1])
                row_split = torch.chunk(column, self.D, dim = 2)
                reduced_block = torch.stack(row_split, dim = 0).reshape(-1, channels, block_dim//self.D, block_dim//self.D).to(self.device[0])
                
                temp = reduced_block.to(self.device[1])
                del column_split, column, row_split
            
        del temp
        
        torch.cuda.empty_cache()

        return reduced_block, channels, block_dim


    def conv_partial_trace(self, tensor, channels, block_dim):
        dimension = block_dim//self.D

        # Kernel is an identity matrix repeated across all the channels
        kernel = torch.eye(dimension, dtype = torch.float).repeat(1, channels, 1, 1).to(self.device[0])
        # Performing the trace
        output_tensor = F.conv2d(tensor, kernel, stride = dimension, padding = 0).to(self.device[0])
        
        del tensor
        del kernel
        torch.cuda.empty_cache()

        return output_tensor
    

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
    
    def partial_trace(self):
        self.log_n_x(self.rho, self.D)

        t1 = time.time()

        reduced_tensor, channels, block_dim = self.block_splitting()
        # dist.barrier()
        reduced_tensor = self.conv_partial_trace(reduced_tensor, channels, block_dim)
        # dist.barrier()
        reduced_tensor = self.matrix_reassembly(reduced_tensor)
        # dist.barrier()

        t2 = time.time()

        total_time = t2-t1

        return reduced_tensor, total_time