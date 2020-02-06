# -*- coding:utf-8 -*-
# @Time: 2020/1/14 9:25
# @Author: jockwang, jockmail@126.com

import pandas as pd
import torch
import logging
from torch_geometric.data import Data

def getGraph(number, dataset='book'):
    df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Graph4CTR/data/'+dataset+'/kg_final.txt', sep='\t', header=None, index_col=None)
    logging.info('Generating subgraph...')
    head_index, tail_index = [], []
    for value in df.values:
        head_index.append(value[0])
        tail_index.append(value[-1])
    edge_index = torch.tensor([head_index, tail_index], dtype=torch.long)
    x = torch.tensor(range(number['a']), dtype=torch.long)
    logging.info('Done.')
    return Data(x=x, edge_index=edge_index)