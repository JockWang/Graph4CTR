# -*- coding:utf-8 -*-
# @Time: 2020/1/14 10:41
# @Author: jockwang, jockmail@126.com
import argparse
import numpy as np
import logging


def process(dataset='book'):
    RATING_FILE_NAME = dict({'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt'})
    SEP = dict({'movie': '::', 'book': ';', 'news': '\t'})
    THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0})
    BASEPATH = '/content/drive/My Drive/Colab Notebooks/Graph4CTR/data/'

    def read_item_index_to_entity_id_file():
        file = BASEPATH + DATASET + '/item_index2entity_id_rehashed.txt'
        logging.info('reading item index to entity id file: ' + file + ' ...')
        i = 0
        for line in open(file, encoding='utf-8').readlines():
            item_index = line.strip().split('\t')[0]
            satori_id = line.strip().split('\t')[1]
            item_index_old2new[item_index] = i
            entity_id2index[satori_id] = i
            i += 1

    def convert_rating():
        file = BASEPATH + DATASET + '/' + RATING_FILE_NAME[DATASET]

        logging.info('reading rating file ...')
        item_set = set(item_index_old2new.values())
        user_pos_ratings = dict()
        user_neg_ratings = dict()

        for line in open(file, encoding='utf-8').readlines()[1:]:
            array = line.strip().split(SEP[DATASET])

            # remove prefix and suffix quotation marks for BX dataset
            if DATASET == 'book':
                array = list(map(lambda x: x[1:-1], array))  # 去掉双引号

            item_index_old = array[1]  # item列
            if item_index_old not in item_index_old2new:  # the item is not in the final item set
                continue
            item_index = item_index_old2new[item_index_old]

            user_index_old = int(array[0])  # 用户

            rating = float(array[2])  # 评分
            if rating >= THRESHOLD[DATASET]:  # 评分阈值
                if user_index_old not in user_pos_ratings:  # 积极评分
                    user_pos_ratings[user_index_old] = set()
                user_pos_ratings[user_index_old].add(item_index)
            else:
                if user_index_old not in user_neg_ratings:  # 消极评分
                    user_neg_ratings[user_index_old] = set()
                user_neg_ratings[user_index_old].add(item_index)

        logging.info('converting rating file ...')
        writer = open(BASEPATH + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
        user_cnt = 0
        user_index_old2new = dict()
        for user_index_old, pos_item_set in user_pos_ratings.items():
            if user_index_old not in user_index_old2new:
                user_index_old2new[user_index_old] = user_cnt
                user_cnt += 1
            user_index = user_index_old2new[user_index_old]

            for item in pos_item_set:
                writer.write('%d\t%d\t1\n' % (user_index, item))
            unwatched_set = item_set - pos_item_set
            if user_index_old in user_neg_ratings:
                unwatched_set -= user_neg_ratings[user_index_old]
            for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
                writer.write('%d\t%d\t0\n' % (user_index, item))
        writer.close()
        logging.info('number of users: %d' % user_cnt)
        logging.info('number of items: %d' % len(item_set))
        return user_cnt, len(item_set)

    def convert_kg():
        logging.info('converting kg file ...')
        entity_cnt = len(entity_id2index)
        relation_cnt = 0

        writer = open(BASEPATH + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

        files = []
        if DATASET == 'movie':
            files.append(open(BASEPATH + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
            files.append(open(BASEPATH + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
        else:
            files.append(open(BASEPATH + DATASET + '/kg_rehashed.txt', encoding='utf-8'))

        for file in files:
            for line in file:
                array = line.strip().split('\t')
                head_old = array[0]
                relation_old = array[1]
                tail_old = array[2]

                if head_old not in entity_id2index:
                    entity_id2index[head_old] = entity_cnt
                    entity_cnt += 1
                head = entity_id2index[head_old]

                if tail_old not in entity_id2index:
                    entity_id2index[tail_old] = entity_cnt
                    entity_cnt += 1
                tail = entity_id2index[tail_old]

                if relation_old not in relation_id2index:
                    relation_id2index[relation_old] = relation_cnt
                    relation_cnt += 1
                relation = relation_id2index[relation_old]

                writer.write('%d\t%d\t%d\n' % (head, relation, tail))

        writer.close()
        logging.info('number of entities (containing items): %d' % entity_cnt)
        logging.info('number of relations: %d' % relation_cnt)
        return entity_cnt

    np.random.seed(555)
    DATASET = dataset
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    user, item = convert_rating()
    all_ = convert_kg()
    logging.info('done')
    return user, item, all_
