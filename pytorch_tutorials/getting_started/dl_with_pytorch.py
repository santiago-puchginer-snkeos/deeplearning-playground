from __future__ import print_function
import torch

# Tensors
a = torch.empty(5, 3)
b = torch.rand(3, 2)
c = torch.zeros(5, 3, dtype=torch.long)
e = c.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
f = torch.randn_like(e, dtype=torch.float)

# Operations
g = torch.rand(6, 3) + torch.rand(6, 3)

# Numpy indexing
print(f[:, 1])

# Resizing
h = g.view(-1, 2)

# Torch to numpy
h_numpy = h.numpy()

# Numpy to torch
h_torch = torch.from_numpy(h_numpy)

# CUDA
if torch.cuda.is_available():
    print('CUDA available')
    device = torch.device('cuda')
    i = torch.ones(3, 3)
    i = i.to(device)   # move tensor to GPU
    j = torch.ones(3, 3, device=device)  # create a tensor directly in GPU
else:
    print('CUDA not available')
