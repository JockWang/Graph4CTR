# -*- coding:utf-8 -*-
# @Time: 2020/1/14 9:08
# @Author: jockwang, jockmail@126.com
# from data import MyDataset
# from graph import getGraph
# import torch
# from model import Model, train
# from processtor import process
# from torch.utils.data import DataLoader
import logging
import argparse

LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def main(dataset='book', mode='GAT', hidden_size=64, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('data processing...')
    user, item, all = process()
    number = {
        'u': user,
        'i': item,
        'a': all,
    }
    logging.info('loading train, test set...')
    data = {
        'train': DataLoader(MyDataset(mode='train', dataset=dataset), batch_size=batch_size, shuffle=True),
        'test': DataLoader(MyDataset(mode='test', dataset=dataset), batch_size=batch_size, shuffle=False),
        'graph': getGraph(number, dataset),
    }
    logging.info('initialization model...')
    i_hidden_list = [100, 16]
    hidden_list = [8, 1]
    model = Model(u_hidden_size=hidden_size, i_hidden_size=hidden_size, number=number,
                  i_hidden_list=i_hidden_list, hidden_list=hidden_list, heads=6,
                  dataset=dataset, mode=mode)
    metrics = ['auc', 'f1']
    train(model=model, data=data, metrics=metrics, epochs=30,
          learning_rate=0.001, weight_decay=0.001, device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='book', help='choose a dataset: book, movie, or others.')
    parser.add_argument('--mode', type=str, default='GAT', help='choose an algorithm of GNN: GCN, GAT, or other.')
    parser.add_argument('--batch', type=int, default=64, help='the batch size.')
    parser.add_argument('--hidden', type=int, default=64, help='the embedding size of user and item.')
    parser.print_help()

    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.mode, batch_size=args.batch, hidden_size=args.hidden)