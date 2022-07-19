# Copyright (c) 2021-2022 Dai HBG

"""
MyDeepModel定义了统一接口的深度模型

日志
2021-11-18
- 初始化
2021-12-07
- 新增自定义NN模型
2022-01-18
- 新增GateNet
"""


import sys
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG')
sys.path.append('C:/Users/HBG/Desktop/Repositories/Daily-Frequency-Quant/QBG')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from Model.Loss import *


class MyDeepModel:
    def __init__(self, input_dim, output_dim=1, loss='ic_mse', degree=2, hinge=0.1, device='cpu'):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param loss: 损失函数类型
        :param degree: 自定义PolyLoss的时候的次数
        :param hinge: 自定义HingeLoss的时候的阈值
        :param device: 训练用

        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.hinge = hinge
        self.device = device

        if loss == 'mse':
            self.loss = MSELoss()
        elif loss == 'mae':
            self.loss = MAELoss()
        elif loss == 'ic':
            self.loss = ICLoss()
        elif loss == 'ic_mse':
            self.loss = ICMSELoss()
        elif loss == 'poly_loss':
            self.loss = PolyLoss(degree=self.degree)
        elif loss == 'hinge_poly_loss':
            self.loss = HingePolyLoss(degree=self.degree, hinge=self.hinge)
        elif loss == 'weighted_poly_loss':
            self.loss = WeightedPolyLoss(degree=self.degree)
        else:
            self.loss = MSELoss()


class NNReg(nn.Module):  # 用于回归的神经网络，默认三层
    def __init__(self, input_dim, output_dim, dropout=0.2, alpha=0.2):
        super(NNReg, self).__init__()
        super().__init__()
        self.input_dim = input_dim  # 输入维度
        self.output_dim = output_dim  # 输出维度
        self.dropout = dropout
        self.alpha = alpha  # LeakyRelu参数

        self.Dense1 = nn.Linear(input_dim, input_dim * 2)
        if input_dim >= 2:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim // 2)
            self.Dense3 = nn.Linear(input_dim // 2, output_dim)
        else:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim)
        self.Dense3 = nn.Linear(input_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        x = self.leakyrelu(self.Dense1(x))
        x = self.leakyrelu(self.Dense2(x))
        x = self.leakyrelu(self.Dense3(x))
        return x


class TSNNReg(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, alpha=0.2, seq_length=5, device='cuda'):
        super(TSNNReg, self).__init__()
        super().__init__()
        self.input_dim = input_dim  # 输入维度
        self.output_dim = output_dim  # 输出维度
        self.dropout = dropout
        self.alpha = alpha  # LeakyRelu参数
        self.seq_length = seq_length
        self.device = device

        self.Dense1 = nn.ModuleList([nn.Linear(input_dim, input_dim * 2).to(device) for i in range(seq_length)])
        if input_dim >= 2:
            self.Dense2 = nn.ModuleList(
                [nn.Linear(input_dim * 2, input_dim // 2).to(device) for i in range(seq_length)])
            self.Dense3 = nn.Linear((input_dim // 2) * seq_length, output_dim).to(device)
        else:
            self.Dense2 = nn.ModuleList([nn.Linear(input_dim * 2, input_dim).to(device) for i in range(seq_length)])
            self.Dense3 = nn.Linear(input_dim * seq_length, output_dim).to(device)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):  # 输入batch_size * seq_length * input_dim的矩阵
        x_1 = [self.leakyrelu(self.Dense1[i](x[:, i, :])) for i in range(self.seq_length)]

        x_2 = torch.cat([self.leakyrelu(self.Dense2[i](x_1[i])) for i in range(self.seq_length)], dim=-1).to(
            self.device)
        x = self.Dense3(x_2)
        return x


class LSTMReg(nn.Module):  # 输入信号的时序值，做回归
    def __init__(self, input_dim, output_dim, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.2):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param hidden_size: 隐藏层维度
        :param num_layers: 隐藏层层数
        :param bidirectional: 是否双向
        :param dropout: dropout值
        """
        super(LSTMReg, self).__init__()
        self.input_dim = input_dim
        self.output_Dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.direction_num = 2 if self.bidirectional else 1  # 双向则置为2
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=self.bidirectional)
        self.Dense1 = nn.Linear(self.hidden_size * self.num_layers * self.direction_num + input_dim, input_dim * 2)
        if input_dim >= 2:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim // 2)
            self.Dense3 = nn.Linear(input_dim // 2, output_dim)
        else:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim)
            self.Dense3 = nn.Linear(input_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):  # 传入batch_size * sequence * input_dim的数据
        output, (h_n, c_n) = self.lstm(x)
        h_n = h_n.transpose(1, 0)
        x = torch.cat([h_n.flatten(start_dim=1), x[:, -1, :]], dim=-1)
        # h_n是(batch_size, num_layers * direction_num, hidden_size)的矩阵
        x = self.leakyrelu(self.Dense1(x))
        x = self.leakyrelu(self.Dense2(x))
        x = self.leakyrelu(self.Dense3(x))
        return x


class Gate(nn.Module):
    def __init__(self, input_dim: int):
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.W = nn.Parameter(torch.zeros(input_dim, input_dim))
        nn.init.xavier_uniform_(self.W.data)
        self.b = nn.Parameter(torch.zeros(input_dim))
        # nn.init.xavier_uniform_(self.b.data)

        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(torch.matmul(x, self.W) + self.b)


class GateNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: int = 0.7, alpha: float = 0.2):
        super(GateNet, self).__init__()
        self.input_dim = input_dim
        self.output_Dim = output_dim

        self.Dense1 = nn.Linear(input_dim, 128)
        self.Dense2 = nn.Linear(128, 64)
        self.Dense3 = nn.Linear(64, 64)
        self.Dense4 = nn.Linear(64, output_dim)

        self.gate0 = Gate(input_dim)
        self.gate1 = Gate(128)
        self.gate2 = Gate(64)
        self.gate3 = Gate(64)

        self.act = nn.LeakyReLU(alpha)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.Dense1(self.gate0(self.dropout(x))))
        x = self.act(self.Dense2(self.gate1(self.dropout(x))))
        x = x + self.dropout(self.gate3(self.act(self.Dense3(self.gate2(self.dropout(x))))))
        x = self.Dense4(x)
        return x


class MyNNModel(MyDeepModel):  # 由于需要统一预测，因此需要额外包装一层NN，在该类里整合方法
    def __init__(self, input_dim, output_dim=1, nn_model=None, loss='ic_mse', degree=2, hinge=0.1, device='cpu'):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param nn_model: 可以传入自定义好的模型
        :param loss: 损失函数类型
        :param degree: 自定义PolyLoss的时候的次数
        :param hinge: 自定义HingeLoss的时候的阈值
        :param device: 训练用
        """
        super(MyNNModel, self).__init__(input_dim=input_dim, output_dim=output_dim, loss=loss, degree=degree,
                                        hinge=hinge, device=device)
        if nn_model is None:
            nn_model = NNReg(input_dim=input_dim, output_dim=output_dim)
        self.model = nn_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

    def fit(self, x, y, epochs=1000):  # 默认传入的x是二维np.array，y是1维度的np.array
        x_train = torch.Tensor(x).to(self.device)
        if len(y.shape) == 2:
            y_train = torch.Tensor(y).to(self.device)
        else:
            y_train = torch.Tensor(y.reshape(-1, 1)).to(self.device)
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(x_train)
            loss = self.loss(out, y_train)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % (epochs // 10) == 0:
                print('epoch {} loss {:.4f}'.format(epoch + 1, float(loss)))
        self.model.eval()

    def predict(self, x):  # 预测
        with torch.no_grad():
            y = self.model(torch.Tensor(x).to(self.device))
            return y[:, 0].cpu().numpy()  # 返回np.array


class MyLSTMModel(MyDeepModel):  # 由于需要统一预测，因此需要额外包装一层NN，在该类里整合方法
    def __init__(self, input_dim, output_dim=1, lstm_model=None, loss='ic_mse',
                 hidden_size=128, num_layers=2, bidirectional=False, dropout=0.2, degree=2, hinge=0.1, device='cpu'):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param lstm_model: 可以自定义时序模型
        :param loss: 指定损失函数
        :param hidden_size: 隐藏层维度
        :param num_layers: 隐藏层层数
        :param bidirectional: 是否双向
        :param dropout: dropout值
        :param degree: 自定义PolyLoss的时候的次数
        :param hinge: 自定义HingeLoss的时候的阈值
        """
        super(MyLSTMModel, self).__init__(input_dim=input_dim, output_dim=output_dim, loss=loss, degree=degree,
                                          hinge=hinge, device=device)
        if lstm_model is None:
            lstm_model = LSTMReg(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size,
                                 num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.model = lstm_model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

    def fit(self, x, y, epochs=1000, batch_size=10000):  # 默认传入的x是三维np.array，为batch_size * sequence_length * input_dim
        x_train = torch.Tensor(x).to(self.device)  # y是1维度的np.array
        if len(y.shape) == 2:
            y_train = torch.Tensor(y).to(self.device)
        else:
            y_train = torch.Tensor(y.reshape(-1, 1)).to(self.device)
        pos = [i for i in range(len(x_train))]
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            x_ = np.random.choice(pos, batch_size)
            out = self.model(x_train[x_])
            loss = self.loss(out, y_train[x_])
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % (epochs // 10) == 0:
                print('loss: {:.4f}'.format(float(loss)))
        self.model.eval()

    def predict(self, x):  # 预测
        with torch.no_grad():
            y = self.model(torch.Tensor(x).to(self.device))
            return y[:, 0].cpu().numpy()  # 返回np.array


class MyTSNNModel:  # 由于需要统一预测，因此需要额外包装一层NN，在该类里整合方法
    def __init__(self, input_dim, output_dim=1, nn_model=None, loss='ic_mse', degree=2, hinge=0.1,
                 device='cpu', seq_length=5):
        super(MyTSNNModel, self).__init__(input_dim=input_dim, output_dim=output_dim, loss=loss, degree=degree,
                                          hinge=hinge, device=device)
        if nn_model is None:
            nn_model = TSNNReg(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length, device=device)
        self.model = nn_model
        self.seq_length = seq_length
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

    def fit(self, x, y, epochs=1000):  # 默认传入的x是三维np.array，y是1维度的np.array
        x_train = torch.Tensor(x).to(self.device)
        if len(y.shape) == 2:
            y_train = torch.Tensor(y).to(self.device)
        else:
            y_train = torch.Tensor(y.reshape(-1, 1)).to(self.device)
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(x_train)
            loss = self.loss(out, y_train)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % (epochs // 10) == 0:
                print('loss: {:.4f}'.format(float(loss)))
        self.model.eval()

    def predict(self, x):  # 预测
        with torch.no_grad():
            y = self.model(torch.Tensor(x).to(self.device))
            return y[:, 0].cpu().detach().numpy()  # 返回np.array


class MyNNDailyModel:  # 由于需要统一预测，因此需要额外包装一层NN，在该类里整合方法
    def __init__(self, input_dim, output_dim=1, nn_model=None, loss='ic_mse', degree=2, hinge=0.1, device='cpu'):
        if nn_model is None:
            nn_model = NNReg(input_dim=input_dim, output_dim=output_dim)
        self.model = nn_model
        super(MyNNDailyModel, self).__init__(input_dim=input_dim, output_dim=output_dim, loss=loss, degree=degree,
                                             hinge=hinge, device=device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

    def fit(self, x, y, epochs=1000):  # 默认传入的x是二维np.array，y是1维度的np.array
        x_train = [torch.Tensor(i).to(self.device) for i in x]
        if len(y[0].shape) == 2:
            y_train = [torch.Tensor(i).to(self.device) for i in y]
        else:
            y_train = [torch.Tensor(i.reshape(-1, 1)).to(self.device) for i in y]
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            loss = 0
            for i in range(len(x_train)):
                out = self.model(x_train[i])
                loss += self.loss(out, y_train[i])
            loss /= len(x_train)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % (epochs // 10) == 0:
                print('loss: {:.4f}'.format(float(loss)))
        self.model.eval()

    def predict(self, x):  # 预测
        with torch.no_grad():
            y = self.model(torch.Tensor(x).to(self.device))
            return y[:, 0].cpu().detach().numpy()  # 返回np.array
