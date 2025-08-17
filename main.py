import torch

import attention as at

print("PyTorch version:", torch.__version__)

t1 = torch.tensor([[1, 2, 3],[4,5,6]])
t2 = torch.tensor([[4, 5, 6], [6,7,8]])

print(torch.transpose(t2,0,1))

print(torch.matmul(t1,torch.transpose(t2,0,1)))


print(t2.ndim)

print(t2.shape)

Q = torch.tensor([[1.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])

# Keys same size as queries
K = torch.tensor([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 1.0]])

# Values also have dimension 3
V = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(at.scaled_dot_product_attention(Q,K,V))




