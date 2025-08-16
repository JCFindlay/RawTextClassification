import torch

print("PyTorch version:", torch.__version__)

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

print("Tensor 1:", t1)
print("Tensor 2:", t2)

output = torch.matmul(t1, t2)
print("Output of matmul:", output)