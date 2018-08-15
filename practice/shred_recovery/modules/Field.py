# coding: utf-8

import random
import numpy as np

from modules.Ant import Ant


class Field(object):

    def __init__(self, cost_matrix, cycle, ant_num, P=5):
        # pheromone 的指数因子
        self.alpha = 1
        # 代价的指数因子
        self.beta = 2
        # pheromone 的衰减系数
        self.decay = 0.5
        # pheromone 的总量
        self.P = P

        # 循环的次数
        self.cycle = cycle
        # 蚂蚁的数量
        self.ant_num = ant_num

        self.__init_field(cost_matrix)

    def __init_field(self, cost_matrix):
        # 位置的数量
        self.node_num = len(cost_matrix)
        # 位置间代价矩阵
        self.cost_matrix = cost_matrix
        # 位置间 pheromone 矩阵
        self.phe_matrix = np.array([[1] * self.node_num] * self.node_num)
        # 最优的代价
        self.best_cost = 1000
        # 最优的路径
        self.best_path = []

    def __cal_prob(self, node):
        """
        计算蚂蚁爬行的可能性列表
        :return:
        """
        cost_ = self.cost_matrix[node]
        prob_ = self.phe_matrix[node] ** self.alpha * (1 / (10 * cost_ + 1)) ** self.beta
        return prob_, cost_

    def __select_from_prob(self, prob_, tabu):
        """
        根据可能性列表选择一个地点作为下一个移动地点
        :param prob_: <np.array> 可能性向量
        :param tabu: <list> 禁忌列表
        :return:
        """
        # 对 prob_ 进行深拷贝，防止对原数据进行修改
        prob_ = prob_.copy()
        # 禁忌表中出现过的位置，再次出现的可能性置为 0
        for node in tabu:
            prob_[node] = 0
        # 赌博轮盘指针
        pointer = random.random() * sum(prob_)
        for index in range(len(prob_)):
            pointer -= prob_[index]
            if pointer <= 0:
                return index

    def __ant_move(self):
        """
        蚂蚁移动方法
        :return: 
        """
        # 初始化一个矩阵用于存储蚂蚁爬过导致 pheromone 的变化
        pheromone_diff = np.array([[0 for x in range(self.node_num)] for y in range(self.node_num)])

        for ant_index in range(self.ant_num):
            ant = Ant(ant_index, self.node_num)
            for move in range(self.node_num - 1):
                prob_, cost_ = self.__cal_prob(ant.node)
                next_node = self.__select_from_prob(prob_, ant.path)
                ant.move(next_node, cost_[next_node])

            # 如果该只蚂蚁爬行路径的代价小于最优代价，则更新最优代价和最优路径
            if ant.total_cost < self.best_cost:
                self.best_cost = ant.total_cost
                self.best_path = ant.path

            # print(ant.total_cost)

            # 计算单位代价的 pheromone 浓度
            pheromone_per_cost = self.P / ant.total_cost
            # 蚂蚁爬行路径上的 pheromone 增加
            for index in range(self.node_num - 1):
                previous = ant.path[index]
                next = ant.path[index + 1]
                pheromone_diff[previous][next] += pheromone_per_cost

        return pheromone_diff

    def __update_pheromone(self, pheromone_diff):
        """
        更新 pheromone
        :param pheromone_diff: <np.array> pheromone 的更新矩阵
        :return:
        """
        # pheromone 随时间衰减
        self.phe_matrix = self.phe_matrix * self.decay
        # 衰减后的 pheromone 叠加上蚂蚁爬过引起的 pheromone 变化
        self.phe_matrix += pheromone_diff
        return self

    def run(self):
        """
        主循环
        :return:
        """
        for cycle in range(self.cycle):
            pheromone_diff = self.__ant_move()
            self.__update_pheromone(pheromone_diff)

        return self.best_path