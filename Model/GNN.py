# Copyright (c) 2022 Dai HBG

"""
GNN

2022-05-22
- init
"""

import torch
import torch.nn as nn

from Model import *


# 总模型
class KGATWithGate(nn.Module):  # 单头注意力
    def __init__(self, raw_input_dim: int, ind_input_dim: int,
                 add_input_dim: int, output_dim: int = 1, hidden_dim: int = 64):
        """
        :param raw_input_dim: 原始特征输入维度
        :param ind_input_dim: 邻居特征输入维度
        :param add_input_dim: 额外用于计算权重维度
        :param output_dim:
        :param hidden_dim: 邻居特征压缩维度
        """
        super(KGATWithGate, self).__init__()

        # 全连接层
        self.Dense1_raw = nn.Linear(raw_input_dim, 764)
        self.Dense1_ind = nn.Linear(ind_input_dim, hidden_dim)
        self.Dense1_add = nn.Linear(add_input_dim, hidden_dim)

        self.Dense2 = nn.Linear(764 + hidden_dim * 3, 256)
        self.Dense3 = nn.Linear(256, output_dim)

        self.Dense3_ind = nn.Linear(hidden_dim, output_dim)
        self.Dense3_add = nn.Linear(hidden_dim, output_dim)

        # dropout层
        self.dropout0_raw = nn.Dropout(0.75)
        self.dropout0_ind = nn.Dropout(0.9)
        self.dropout0_add = nn.Dropout(0.7)

        self.dropout1 = nn.Dropout(0.65)
        self.dropout2 = nn.Dropout(0.55)

        # gate层
        self.gate0_raw = Gate(raw_input_dim)
        self.gate0_ind = Gate(ind_input_dim)
        self.gate0_add = Gate(add_input_dim)

        self.gate1 = Gate(768 + hidden_dim * 3)
        self.gate2 = Gate(256)

        # 激活层
        self.act = nn.LeakyReLU(0.2)

        # 权重计算层
        self.gat = GAT(raw_intput_dim=768, add_intput_dim=add_input_dim, output_dim=hidden_dim)

    def forward(self, x_raw, x_ind, x_add, ind):
        """
        :param x_raw:
        :param x_ind:
        :param x_add:
        :param ind: 邻接矩阵
        """
        # 过第一层
        x_raw = self.Dense1_raw(self.gate0_raw(self.dropout0_raw(x_raw)))
        x_ind = self.Dense1_ind(self.gate0_ind(self.dropout0_ind(x_ind)))
        x_add_1 = self.Dense1_add(self.gate0_add(self.dropout0_add(x_add)))

        # 两个输出
        output_ind = self.Dense3_ind(self.act(x_ind))
        output_add = self.Dense3_add(self.act(x_add_1))

        # 计算原始数据的邻居加权，注意需要实验在哪一层进行聚合
        x_gat = self.gat(x_raw, x_add, ind)

        x = torch.cat([x_raw, x_ind, x_add_1, x_gat], dim=1)
        x = self.Dense2(self.act(self.gate1(self.dropout1(x))))
        x = self.Dense3(self.act(self.gate2(self.dropout2(x))))

        return x, output_ind, output_add


class GAT(nn.Module):  # 接受原始特征和辅助特征计算原始特征的联合权重
    def __init__(self, raw_input_dim: int, add_input_dim: int, output_dim: int):
        super(GAT, self).__init__()
        # 使用一个双层的GateNet来计算权重
        self.Dense1 = nn.Linear(raw_intput_dim + add_intput_dim, 32)
        self.left1 = nn.Parameter(torch.zeros((32, 1)))
        self.right1 = nn.Parameter(torch.zeros((32, 1)))
        nn.init.xavier_uniform(self.left1.data)
        nn.init.xavier_uniform(self.right1.data)

        # 输出层，用一个线性变换
        self.Dense_output = nn.Linear(raw_input_dim, output_dim)

        # dropout和激活函数
        self.dropout = nn.Dropout(0.6)
        self.act = nn.LeakyReLU(0.2)

        # gate
        self.gate1 = Gate(raw_input_dim + add_input_dim)
        self.gate2 = Gate(raw_input_dim + add_input_dim)

    def forward(self, x_raw, x_ind, ind):
        """
        :param x_raw:
        :param x_ind:
        :param ind: 邻接矩阵，由于目前
        """
        x = torch.cat([x_raw, x_ind], dim=1)  # 先拼接起来
        h = self.act(self.Dense1(self.gate1(self.dropout(x))))
        a_left = torch.matmul(h, self.left1)
        a_right = torch.matmul(h, self.right1)
        e = self.act(a_left.T + a_right)  # 未归一化权重

        z = -1e12 * torch.ones_like(e)
        att = torch.where(ind > 0, e, z)
        att = F.softmax(att, dim=1)  # 归一化权重

        h_ = self.act(self.Dense_output(self.gate2(self.dropout(x))))
        return torch.matmul(att, h_)
