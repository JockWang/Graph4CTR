# -*- coding:utf-8 -*-
# @Time: 2020/1/14 9:13
# @Author: jockwang, jockmail@126.com

from torch.utils.data import Dataset
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, mode='train', dataset='book'):
        super(MyDataset, self).__init__()
        df = pd.read_csv('data/'+dataset+'/ratings.txt', sep='\t', header=None, index_col=None).values
        train, test = train_test_split(df, test_size=0.2, random_state=2019)
        if mode == 'train':
            self.data = train
        else:
            self.data = test
        logging.info(mode+' set size:'+str(self.data.shape[0]))

    def __getitem__(self, index):
        temp = self.data[index]
        return torch.tensor(temp[0], dtype=torch.long), torch.tensor(temp[1], dtype=torch.long), torch.tensor([temp[2]], dtype=torch.float)

    def __len__(self):
        return len(self.data)