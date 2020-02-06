# -*- coding:utf-8 -*-
# @Time: 2020/1/14 9:13
# @Author: jockwang, jockmail@126.com

from torch.utils.data import Dataset
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class MyDataset(Dataset):
    def __init__(self, mode='train', item_size=0, dataset='book'):
        super(MyDataset, self).__init__()
        df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Graph4CTR/data/' + dataset + '/ratings_final.txt',
                         sep='\t', header=None, index_col=None).values
        train, test = train_test_split(df, test_size=0.2, random_state=2019)
        self.item_size = item_size
        if mode == 'train':
            self.data = train
        else:
            self.data = test
        logging.info(mode + ' set size:' + str(self.data.shape[0]))

    def __getitem__(self, index):
        temp = self.data[index]
        item = np.zeros(shape=(1, self.item_size))
        item[0, temp[1]] = 1
        return torch.tensor(temp[0], dtype=torch.long), torch.tensor(item, dtype=torch.float), torch.tensor(
            [temp[2]], dtype=torch.float)

    def __len__(self):
        return len(self.data)
