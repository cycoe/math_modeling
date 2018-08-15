# coding: utf-8

import random


class Ant(object):

    def __init__(self, id, site_num):
        # 每只蚂蚁的 id
        self.id = id
        # 能够到达的目的地总数
        self.site_num = site_num
        self.__init_data()

    def __init_data(self):
        """
        初始化蚂蚁的数据
        :return:
        """
        # 蚂蚁爬行的路径记录数组
        self.path = []
        # 蚂蚁爬行的总距离
        self.total_cost = 0.0
        # 蚂蚁的移动次数
        self.move_count = 0
        # 蚂蚁当前的位置
        self.node = -1

        # 随机选取一个蚂蚁的起始位置
        node = random.randint(0, self.site_num - 1)
        # 更新蚂蚁的位置
        self.move(node, 0)

    def move(self, node, cost):
        """
        蚂蚁的移动方法
        :param node: 要到达的位置
        :return:
        """
        self.node = node
        self.path.append(node)
        self.move_count += 1
        self.total_cost += cost