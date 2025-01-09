"""
Exercise link: https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises
"""

import torch
# Exercise 2
t2 = torch.rand(7,7)

# Exercise 3
t3 = torch.rand(1,7)
mul1 = torch.matmul(t3, t2)  # First way
mul2 = t3 @ t2  # Second way

# Exercise 4
torch.manual_seed(0)
t4_1 = torch.rand(7,7)
torch.manual_seed(0)
t4_2 = torch.rand(1,7)
mul_4 = t4_2 @ t4_1

# Exercise 5
torch.cuda.manual_seed(1234)

# Exercise 6
torch.manual_seed(1234)
t6_1 = torch.rand(2,3).cuda()
torch.manual_seed(1234)
t6_2 = torch.rand(2,3).cuda()

# Exercise 7
mul7 = t6_1 @ t6_2.mT

# Exercise 8
max8 = torch.max(mul7)
min8 = torch.min(mul7)

# Exercise 9
max9 = torch.argmax(mul7)
min9 = torch.argmin(mul7)

# Exercise 10
torch.manual_seed(7)
tensor1 = torch.rand(1, 1, 1, 10)
tensor2 = tensor1.squeeze()
print("Tensor1:", tensor1)
print("Shape of Tensor1:", tensor1.shape)
print("Tensor2:", tensor2)
print("Shape of Tensor2:", tensor2.shape)