# @Time: 2024/10/4 12:06
# @Author: xy

import torch
import torch.nn.functional as F
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 使用内置的 GELU
output = torch.nn.functional.gelu(x)
print("GELU Output (PyTorch):", output)


class MyGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # GELU 前向计算
        ctx.save_for_backward(x)
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        # erf(x) = 2/sqrt(pi) * inf_{0}^(x) exp(-0.5*t**2)dt

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # GELU 的导数
        gelu_derivative = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) + \
                          (x / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * torch.exp(-0.5 * x ** 2)))
        return grad_output * gelu_derivative


if __name__ == '__main__':
    input = torch.tensor([1, 2, 3], dtype=torch.float)
    mlp = nn.Linear(in_features=3, out_features=1, bias=True)
    output = mlp(input)
    output = MyGelu.apply(output)  # do not call MyGelu().forward for running efficiency
    print(output)
