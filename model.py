# -*- coding:utf-8 -*-
# @Time: 2020/1/14 9:10
# @Author: jockwang, jockmail@126.com

import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch
import logging
from sklearn.metrics import roc_auc_score, f1_score

class Model(nn.Module):
    def __init__(self, u_hidden_size, i_hidden_size, number, i_hidden_list, hidden_list,
                 heads=6, dataset='movie', mode='GAT'):
        super(Model, self).__init__()
        self.u_hidden_size, self.i_hidden_size = u_hidden_size, i_hidden_size
        self.u_nodes, self.i_nodes = number['u'], number['a']
        self.u_embedding = nn.Embedding(self.u_nodes, self.u_hidden_size)
        self.i_embedding = nn.Embedding(self.i_nodes, self.i_hidden_size)
        self.convs = self.ModuList()
        i_hidden_list = [i_hidden_size] + i_hidden_list
        if mode == 'GCN':
            self.convs = self.ModuList([GCNConv(i_hidden_list[i-1], i_hidden_list[i])
                         for i in range(1, len(i_hidden_list))])
        elif mode == 'GAT':
            self.convs = self.ModuList([GATConv(i_hidden_list[i-1], i_hidden_list[i], heads=heads, concat=False)
                         for i in range(1, len(i_hidden_list))])
        hidden_list = [i_hidden_list[-1] + u_hidden_size] + hidden_list
        self.liners = nn.ModuleList([nn.Linear(hidden_list[i-1], hidden_list[i])
                                     for i in range(1, len(hidden_list))])
        if hidden_list[-1] == 1:
            self.final = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:
            self.final = nn.Softmax()
            self.loss = nn.CrossEntropyLoss()

    def forward(self, user, item, graph):
        u_emb = self.u_embedding(user)
        i_emb = self.i_embedding(item)

        for layer in self.convs:
            i_emb = layer(i_emb)
        out = torch.cat([u_emb, i_emb])
        for layer in self.liners:
            out = layer(out)

        return self.final(out)

def evaluate(model, data, graph, metrics, device='cup'):
    y, y_ = [], []
    for k, [user, item, label] in enumerate(data['test']):
        u, i, l = user.to(device), item.to(device), label

        y += label.tolist()[0]
        out = model(u, i, graph)
        y_ += out.tolist()[0]
    line = ''
    for metric in metrics:
        if metric == 'auc':
            line += (' ' + metric + ': %.6f' % roc_auc_score(y, y_))
        if metric == 'f1':
            line += (' ' + metric + ': %.6f' % f1_score(y, y_))
    logging.info(line)

def train(model, data, metrics, epochs=20, learning_rate=0.01, weight_decay=0.01, device='cpu'):
    model.to(device)
    graph = data['graph'].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        running_loss = 0
        for k, [user, item, label] in enumerate(data['train']):
            u, i, l = user.to(device), item.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(u, i, graph)
            loss = model.loss(out, l)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if k % 100 == 99:
                logging.info('Epoch:%d Step:%d loss:%.5f' % (epoch + 1, k+1, running_loss / 100))
                running_loss = 0
            if k % 100 == 499:
                evaluate(model, data['test'], graph, metrics, device)
    logging.info('Final evaluation:')
    evaluate(model, data['test'], graph, metrics, device)