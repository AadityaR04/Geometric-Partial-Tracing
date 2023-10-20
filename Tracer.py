import torch

def trace(loader, output, device, Partial_Trace):
    for batch in loader:
        batch = batch.to(device)
        out = Partial_Trace(batch)
        output.append(out)
    
    reduced_tensor = torch.cat(output, dim = 0)
    
    return reduced_tensor