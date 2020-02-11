# -*- coding:utf-8 -*-
# @Time: 2020/2/6 15:12
# @Author: jockwang, jockmail@126.com
import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from utils import PoincareBall

class HNN(nn.Module):
    def __init__(self, in_features, out_features, c, dropout=0, act=F.relu, use_bias=False):
        super(HNN, self).__init__()
        self.manifold = PoincareBall()
        self.c = nn.Parameter(torch.Tensor([c]))
        self.in_channels = in_features
        self.out_channels = out_features
        self.dropout = dropout
        self.act = act
        self.use_bias = use_bias

        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化weight、bias
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        x = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            x = self.manifold.mobius_add(x, hyp_bias, c=self.c)
            x = self.manifold.proj(x, self.c)
        xt = self.act(self.manifold.logmap0(x, c=self.c))
        xt = self.manifold.proj_tan0(xt, c=self.c)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c), c=self.c)

class HGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, c_in=1., c_out=1.):
        super(HGCN, self).__init__(aggr='add')
        self.manifold = PoincareBall()
        self.c_in = nn.Parameter(torch.Tensor([c_in]))
        self.c_out = nn.Parameter(torch.Tensor([c_out]))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels))

        self.dropout = 0
        self.use_bias = False
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels))

        self.act = F.leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        # init weight、bias、att.
        glorot(self.att)
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        mv = self.manifold.mobius_matvec(self.weight, x, self.c_in)
        x = self.manifold.proj(mv, self.c_in)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            x = self.manifold.mobius_add(x, hyp_bias, c=self.c_in)
            x = self.manifold.proj(x, self.c_in)
        x = self.propagate(edge_index, size=size, x=x)

        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def message(self, edge_index_i, x_i, x_j, size_i):
        x_j = x_j.view(-1, self.out_channels)
        if x_i is None:
            weight = (x_j * self.att[:, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.out_channels)
            weight = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        weight = softmax(weight, edge_index_i, size_i)

        return x_j * weight.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return 'HGCN inputs={}, outputs={}'.format(
            self.in_channels, self.out_channels
        )
