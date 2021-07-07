import torch

input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.randint(5, (3,), dtype=torch.int64)
print(target)
# loss = F.cross_entropy(input, target)

