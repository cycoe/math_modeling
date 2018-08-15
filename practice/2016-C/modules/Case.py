#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Case(object):

    """Case to handle terminals and stations"""

    # 光速
    C = 299792500

    def __init__(self):
        # number of stations
        self.num_station = 0
        # number of terminals
        self.num_terminal = 0
        # coordinate matrix of stations
        self.station_coordinate = []
        # coordinate matrix of terminals
        self.terminal_coordinate = []
        # samples
        self.samples = []

    def load_case(self, path):
        """
        load data from file
        """
        with open(path) as fp:
            content = fp.readlines()
            self.num_station = self._get_number(content[0])
            self.num_terminal = self._get_number(content[1])
            self.station_coordinate = self._get_matrix(content[3: 33])
            self.samples = self._get_matrix(content[33:]) * Case.C
        return self

    def init_argvs(self):
        # K 为每个基站坐标矢量分量平方和的矩阵
        self.K = np.reshape(np.sum(self.station_coordinate ** 2, axis=1),
                            (self.num_station, 1))
        # c 为各基站坐标与 1 号基站坐标的差值矩阵
        self.c = self.station_coordinate - self.station_coordinate[0]
        # r 为终端距离各个基站与 1 号基站的差值矩阵
        # 在 numpy 中，一个一维的 array 对象会自动退化为向量，因此需要使用 reshape 重新生成矩阵
        self.r = self.samples - \
            np.reshape(self.samples[:, 0], (self.num_terminal, 1))

    def first(self):
        self.terminal_coordinate = []
        for index in range(self.num_terminal):
            # 计算矩阵中重复用到的 [r2,1 r3,1,...rn,1]
            r = np.reshape(self.r[index, 1:], (self.num_station - 1, 1))
            # 合并出系数矩阵 G
            G = -np.hstack((self.c[1:], r))
            # 计算右侧非线性部分
            h = 1 / 2 * (r ** 2 - self.K[1:] + self.K[0])
            # 将数组转换为矩阵方便矩阵运算
            G = np.mat(G)
            h = np.mat(h)
            Z0 = (G.T * G).I * G.T * h
            # 重新将 Z0 转化成数组，方便后面计算
            Z0 = np.array(Z0)

            # 根据上一步计算得到的终端坐标重新计算距离矩阵
            d = np.sum((self.station_coordinate - np.reshape(Z0[0:3], (1, 3)))
                       ** 2, axis=1) ** 0.5
            # 重新计算 r
            r = np.reshape(d[1:] - Z0[3], (self.num_station - 1, 1))
            # 重新计算 G
            G = -np.hstack((self.c[1:], r))
            # 重新计算 h
            h = 1 / 2 * (r ** 2 - self.K[1:] + self.K[0])
            G = np.mat(G)
            h = np.mat(h)
            # 创建对角矩阵 B
            B = np.mat(np.diag(d[1:]))
            # 计算 psi
            psi = B * B
            # 重新计算 Z0
            Z0 = (G.T * psi.I * G).I * G.T * psi.I * h

            self.terminal_coordinate.append(np.array(Z0))
        return np.array(self.terminal_coordinate)

    def second(self):
        pass

    def _get_number(self, row):
        """
        read a row as a number
        """
        return int(row.strip())

    def _get_matrix(self, lines, sep='\t'):
        """
        read rows as matrix
        """
        return np.loadtxt(lines)
