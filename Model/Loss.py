# Copyright (c) 2021 Dai HBG

"""
该代码定义了深度模型的不同loss
开发日志
2021-11-18
-- 初始化
2021-11-22
-- 新增HuberLoss和QuantileLoss
"""


import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        super().__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))


class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()
        super().__init__()

    def forward(self, x, y):
        x -= torch.mean(x)
        y -= torch.mean(y)
        return -torch.mean(x * y) / (torch.std(x) * torch.std(y))


class ICMSELoss(nn.Module):
    def __init__(self, mse_coef=1):
        super(ICMSELoss, self).__init__()
        super().__init__()
        self.mse_coef = mse_coef

    def forward(self, x, y):
        mseloss = torch.mean((x - y) ** 2)
        x -= torch.mean(x)
        y -= torch.mean(y)
        return torch.mean(x * y) / (torch.std(x) * torch.std(y)) + self.mse_coef * mseloss


class PolyLoss(nn.Module):  # 不同次数的损失
    def __init__(self, degree=3):
        super(PolyLoss, self).__init__()
        self.degree = degree

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y) ** self.degree)


class HingePolyLoss(nn.Module):  # 不同次数的HingeLoss
    def __init__(self, degree=2, hinge=0.1):
        """
        :param degree: 损失次数
        :param hinge: 损失阈值
        """
        super(HingePolyLoss, self).__init__()
        self.degree = degree
        self.hinge = hinge

    def forward(self, x, y):
        z = torch.abs(x - y)
        z[z < self.hinge] = 0
        return torch.mean(z ** self.degree)


class WeightedPolyLoss(nn.Module):  # 用y加权的PolyLoss，此时必须要求y是关于0对称的，也就是标准化y
    def __init__(self, degree=2):
        super(WeightedPolyLoss, self).__init__()
        self.degree = degree

    def forward(self, x, y):
        return torch.mean((torch.abs(x - y) ** self.degree) * torch.abs(y))


class HuberLoss(nn.Module):  # HuberLoss
    def __init__(self, degree_1=2, degree_2=1, hinge=0.1):
        super(HuberLoss, self).__init__()
        self.degree_1 = degree_1
        self.degree_2 = degree_2
        self.hinge = hinge
        self.adj = hinge ** (degree_1 / degree_2) - hinge

    def forward(self, x, y):
        z_1 = torch.abs(x - y)
        z_1[z_1 > self.hinge] = 0
        z_2 = torch.abs(x - y)
        z_2[z_2 <= self.hinge] = 0
        return torch.mean(z_1**self.degree_1 + (z_2+self.adj)**self.degree_2)
