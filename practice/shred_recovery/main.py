#!/usr/bin/python3.6
# coding: utf-8

#
#                            _ooOoo_
#                           o8888888o
#                           88" . "88
#                           (| -_- |)
#                           O\  =  /O
#                        ____/`---'\____
#                      .'  \\|     |//  `.
#                     /  \\|||  :  |||//  \
#                    /  _||||| -:- |||||-  \
#                    |   | \\\  -  /// |   |
#                    | \_|  ''\---/''  |   |
#                    \  .-\__  `-`  ___/-. /
#                  ___`. .'  /--.--\  `. . __
#               ."" '<  `.___\_<|>_/___.'  >'"".
#              | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#              \  \ `-.   \_ __\ /__ _/   .-` /  /
#         ======`-.____`-.___\_____/___.-`____.-'======
#                            `=---='
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                      Buddha Bless, No Bug !

import os
import time
import glob
import numpy as np

from modules.Array import Array
from modules.Field import Field


def timer(func):
    """
    计时装饰器
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('function <{}> running for {} seconds'.format(func.__name__, str(end - start)))
        return result
    return wrapper

def reset(path):
    """
    以列表中最大的元素的位置为中心，将左右两边的元素对调位置
    针对路径环，从最大元素的位置展开成路径，并将最大元素作为起始点
    :param path: <list> 原始 path 列表
    :return: <list> 重排后的 path 列表
    """
    # path 中最大元素的索引
    pointer = path.index(len(path) - 1)
    # 将 pointer 前的元素移至列表后面
    path.extend(path[:pointer])
    del path[:pointer + 1]
    return path

def sort(array_, direction=Array.RIGHT, cycle=20, ant_num=400, P=5):
    """
    排序方法
    :param array_: <list of Array> 需要排序的 Array 对象的列表
    :param direction: Array 相邻对象比较的方向
    :param cycle: 蚁群算法的总迭代次数
    :param ant_num: 蚂蚁的总数
    :return: <Array> 合并后的 Array 对象
    """
    # 目的地的数量
    site_num = len(array_)
    # 目的地之前的代价矩阵
    cost_matrix = 1 - np.array(
        [[array_[y].match(array_[x], direction) for x in range(site_num)] for y in range(site_num)])
    # 初始化 Field 对象用于处理蚁群算法
    field = Field(cost_matrix, cycle, ant_num, P)
    # 运行蚁群算法获得最佳路径
    best_path = field.run()
    # 根据定义的起始点对路径进行重排
    best_path = reset(best_path)

    # 初始化 Array 对象用于储存最终合并得到的矩阵
    array_join = Array()
    # 根据矩阵比较的方向，使 join_method 指向对应方向的合并方法
    if direction == Array.RIGHT:
        join_method = array_join.hjoin
    elif direction == Array.BOTTOM:
        join_method = array_join.vjoin
    else:
        join_method = array_join.hjoin

    # 按照路径顺序合并矩阵
    for index in range(site_num - 1):
        join_method(array_[best_path[index]])

    return array_join

def in_inner_list(item, item_list):
    """
    判断 item 是否在列表内的列表里
    :param item: 需要判断的对象
    :param item_list: <list of list of item>
    :return:
    """
    for item_ in item_list:
        # 若 item 在其中一个列表 item_ 中
        # 则返回 item_
        if item in item_:
            return item_
    # 不存在则返回 False
    return False

def get_longests(item_list, num):
    """
    获得列表中最长的几个元素列表
    :param item_list: <list> 待处理的元素列表
    :param num: <int> 取前 num 个最长元素
    :return: <list> 筛选过的元素列表
    """
    # 列表总长
    total_num = len(item_list)
    # 元素长度列表
    length_ = [len(item) for item in item_list]

    # 如果列表的长度不比 num 大，则无需处理
    if total_num <= num:
        return item_list

    for cycoe in range(total_num - num):
        index = length_.index(min(length_))
        del item_list[index]
        del length_[index]

    return item_list

def cluster(dis_matrix):
    """
    聚类方法
    :param dis_matrix: 元素距离矩阵
    :return: <list> 分类列表
    """
    # 初始化分类列表
    class_list = []
    dis_matrix = dis_matrix
    # 将矩阵对角线上的值设为 10
    for i in range(len(dis_matrix)):
        dis_matrix[i][i] = 10

    while True:
        # 从距离矩阵中取出最小的元素索引
        position = dis_matrix.argmin()
        # 取出的索引是按一维矩阵计算
        # 需要将索引转化成 x 和 y 坐标
        y = position // dis_matrix.shape[1]
        x = position % dis_matrix.shape[1]

        # 获取 x 和 y 是否在分类列表中存在
        # 若存在则得到所在的分类
        x_in = in_inner_list(x, class_list)
        y_in = in_inner_list(y, class_list)

        # 如果 x 在分类中而 y 不在
        # 并且 x 所在分类的列表没满
        # 将 y 也加入 x 的分类
        if x_in and not y_in:
            if len(x_in) < 19:
                x_in.add(y)
            # else:
            #     dis_matrix[y] = 10
            #     dis_matrix[:, y] = 10

        # 如果 y 在分类中而 x 不在
        # 并且 y 所在分类的列表没满
        # 将 x 也加入 y 的分类
        elif not x_in and y_in:
            if len(y_in) < 19:
                y_in.add(x)
            # else:
            #     dis_matrix[x] = 10
            #     dis_matrix[:, x] = 10

        # 如果 x 和 y 都不在分类中
        # 则将 x, y 放入新分类
        elif not x_in and not y_in:
            class_list.append({x, y})

        # 如果 x 和 y 都已经在存在的分类中
        # 并且 x 和 y 不在同一类
        # 并且 x 的分类和 y 的分类合并后长度没有超限
        # 则将 x 的分类和 y 的分类合并
        elif x_in != y_in and len(x_in) + len(y_in) <= 19:
            for item in y_in:
                x_in.add(item)
            del class_list[class_list.index(y_in)]

        dis_matrix[x][y] = 10
        dis_matrix[y][x] = 10

        if dis_matrix.min() == 10:
            break

    return class_list


def plot(vector, color='r', label=''):
    import matplotlib.pyplot as plt

    plt.plot(vector, color, label=label)
    plt.xlabel('pixel')
    plt.ylabel('character density')
    plt.title('character density distribution')
    plt.legend()
    plt.savefig('res/{}.jpg'.format(label))
    plt.close('all')

@timer
def ants_algorithm_h():
    array_pool = []
    problem = 2
    for image_path in glob.glob('res/attachments/{}/*.bmp'.format(str(problem))):
        array = Array().load_image(image_path)
        array_pool.append(array)
    array_pool.append(Array().load_array([[255] * 1] * array_pool[0].height))

    array_join = sort(array_pool, Array.RIGHT, 20, 20)
    array_join.convert_to_image('problem{}.jpg'.format(str(problem)))

@timer
def ants_algorithm_h_v():
    array_pool = []
    problem = 3
    # 将所有的矩阵加入到矩阵池中
    for image_path in glob.glob('res/attachments/{}/*.bmp'.format(str(problem))):
        array = Array().load_image(image_path)
        array_pool.append(array)


    site_num = len(array_pool)
    # 根据行特征算法获得各个矩阵的距离矩阵
    # if os.path.exists('dis_matrix.npy'):
    #     dis_matrix = np.load('dis_matrix.npy')
    # else:
        # dis_matrix = np.array([[np.sum(np.abs(array_pool[y].get_row() - array_pool[x].get_row())) / 10 for x in range(site_num)] for y in range(site_num)])
    dis_matrix = 1 - np.array(
        [[array_pool[y].get_row().match(array_pool[x].get_row(), Array.RIGHT) for x in range(site_num)] for y in range(site_num)])
        # np.save('dis_matrix.npy', dis_matrix)

    print(dis_matrix)

    # 聚类
    class_list = cluster(dis_matrix)
    new_class_list = get_longests(class_list, 11)
    print(new_class_list)

    # 绘图
    # plot(array_pool[26].get_row().image_array, 'r', 'image 26')
    # plot(array_pool[165].get_row().image_array, 'g', 'image 165')
    # plot(array_pool[204].get_row().image_array, 'b', 'image 204')

    # 根据分类，先沿横向合并，再沿纵向合并
    array_ver = []
    for class_ in new_class_list:
        array_ = []
        for index in class_:
            array_.append(array_pool[index])
        array_.append(Array().load_array([[255] * 1] * array_[0].height))
        array_ver.append(sort(array_, Array.RIGHT, cycle=20, ant_num=200, P=5))

    array_ver.append(Array().load_array([[255] * array_ver[0].width] * 1))
    sort(array_ver, Array.BOTTOM, cycle=20, ant_num=400, P=3).convert_to_image('problem{}.jpg'.format(str(problem)))

if __name__ == '__main__':
    ants_algorithm_h_v()