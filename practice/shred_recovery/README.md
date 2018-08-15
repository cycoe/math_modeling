## 碎纸复原问题

### 问题分析

根据题意，将一张印有文字的纸沿纵向等宽切割成 19 张条形纸片，并打乱顺序。设计一个算法将打乱顺序的纸片按正确的顺序排列，并复原。该问题是一个 TSP（旅行商）问题，属于组合优化问题，被证明具有 NPC 计算复杂性。TSP 问题可以利用枚举法得到精确解，但当问题的维度增大时，会导致组合爆炸。因此考虑使用近似算法，常用的近似算法有贪婪算法、模拟退火算法、遗传算法、蚁群算法等。针对此问题，我们使用蚁群算法进行处理。

### 一维切割的情况

#### 问题的形式化描述

利用 python 的 PIL 模块读取附件 1 中的图片，灰度化后转换为矩阵。$S_i$为第 $i$ 个纸片对应的灰度矩阵，形式如下
$$
S_i = \left[
\begin{matrix}
0 & 0 & \cdots & 0\\
255 & 0 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 255 & \cdots & 0\\
\end{matrix}
\right]
$$
矩阵中的每个元素的取值范围为 $a_{ij}\in \left[0, 255\right]$。定义所有纸片灰度矩阵的集合为 $\mathbb{S} = \left\{S_1, S_2, \cdots, S_{19}\right\}$。该问题中，定义 $R_{origin}$ 为未分割前完整纸片的灰度矩阵，$R_{a_1, a_2, \cdots, a_{19}} = \left[\begin{matrix}S_{a_1} & S_{a_2} & \cdots & S_{a_{19}}\end{matrix}\right]$ 为以任意顺序拼接的得到的灰度矩阵。

#### 问题的解

$S_1, S_2, \cdots, S_{19}$ 的一个最短的 Hamilton 回路（有向环）就是该问题的一个解，$S_1, S_2, \cdots, S_{19}$ 的所有的 Hamilton 回路就构成了问题的解空间。由此，该问题的时间复杂度为 $O(n!)$。以矩阵的形式表达，则矩阵 $R_{origin}$ 即为问题的一个解，$R_{a_1, a_2, \cdots, a_{19}}$ 构成的集合 $\mathbb{R}$ 即为问题的搜索空间，求解的过程就是在 $\mathbb{R}$ 中搜索 $R_{origin}$ 的过程。

#### 蚁群算法

>  In [computer science](https://en.wikipedia.org/wiki/Computer_science) and [operations research](https://en.wikipedia.org/wiki/Operations_research), the **ant colony optimization** [algorithm](https://en.wikipedia.org/wiki/Algorithm) (**ACO**) is a [probabilistic](https://en.wikipedia.org/wiki/Probability) technique for solving computational problems which can be reduced to finding good paths through [graphs](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)). **Artificial Ants** stand for [multi-agent](https://en.wikipedia.org/wiki/Multi-agent) methods inspired by the behavior of real ants. The pheromone-based communication of biological [ants](https://en.wikipedia.org/wiki/Ant) is often the predominant paradigm used.[[2\]](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#cite_note-2) Combinations of Artificial Ants and [local search](https://en.wikipedia.org/wiki/Local_search_(optimization)) algorithms have become a method of choice for numerous optimization tasks involving some sort of [graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)), e. g., [vehicle routing](https://en.wikipedia.org/wiki/Vehicle_routing_problem) and Internet [routing](https://en.wikipedia.org/wiki/Routing). The burgeoning activity in this field has led to conferences dedicated solely to Artificial Ants, and to numerous commercial applications by specialized companies such as [AntOptima](https://en.wikipedia.org/w/index.php?title=AntOptima&action=edit&redlink=1). As an example, [Ant colony optimization](https://en.wikipedia.org/wiki/Ant_colony_optimization)[[3\]](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#cite_note-3) is a class of [optimization](https://en.wikipedia.org/wiki/Optimization_(computer_science)) [algorithms](https://en.wikipedia.org/wiki/Algorithm) modeled on the actions of an [ant colony](https://en.wikipedia.org/wiki/Ant_colony). Artificial 'ants' (e.g. simulation agents) locate optimal solutions by moving through a [parameter space](https://en.wikipedia.org/wiki/Parameter_space) representing all possible solutions. Real ants lay down [pheromones](https://en.wikipedia.org/wiki/Pheromone) directing each other to resources while exploring their environment. The simulated 'ants' similarly record their positions and the quality of their solutions, so that in later simulation iterations more ants locate better solutions.[[4\]](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#cite_note-4) One variation on this approach is [the bees algorithm](https://en.wikipedia.org/wiki/Bees_algorithm), which is more analogous to the foraging patterns of the [honey bee](https://en.wikipedia.org/wiki/Honey_bee), another social insect.

```python
for cycle in cycles:
	for ant in ants:
        select the most probably node to move
        move ant to next node
        record the pheromone that the ant left
    update pheromone
```

$$
\Delta \tau^k_{ij} = \dfrac{P}{\sum L_k}
$$

$$
\tau_{ij}(n+1) = \rho \times \tau_{ij}(n) + \sum_{k=1}^m \Delta \tau^k_{ij}
$$


$$
p^k_{ij} = \dfrac{{\tau_{ij}}^\alpha {\eta_{ij}}^\beta}{\sum_{j\in \Lambda} {\tau_{ij}}^\alpha {\eta_{ij}}^\beta}
$$

$$
\eta_{ij} = \dfrac{1}{d_{ij}}
$$

$d_{ij}$ 为节点 $i$ 与 节点 $j$ 之前的距离，$\sum L_k$ 为第 $k$ 只蚂蚁爬过的路径总长，$P$ 为每只蚂蚁留下的 pheromone 总量，$\Delta \tau^k_{ij}$ 为第 $k$ 只蚂蚁爬过后对节点 $i$ 到 $j$ 之间路径 pheromone 浓度的改变，$\rho$ 是一个衰减系数，表征每次循环后 pheromone 浓度的减小，$\tau_{ij}(n)$ 表征第 $n$ 次循环时 $i$ 对 $j$ 的 pheromone 浓度，$p^k_{ij}$ 表征第 $k$ 只蚂蚁从 $i$ 爬到 $j$ 的可能性。

##### 节点间的距离

蚁群算法中，蚂蚁从 $i$ 到 $j$ 的可能性与 $i$ 到 $j$ 的距离和 $i$ 到 $j$ 的 pheromone 浓度有关，因此需要一种表征 $d_{ij}$ 的方法。

对于两张纸片，我们认为它们边缘重合的越好，则它们之间的距离越相近。定义纸片 $i$ 与纸片 $j$ 之间的相似度为 $m_{ij}$，则定义 $d_{ij} = 1 - m_{ij}$。

|      符号       |             含义             |  numpy 调用  |
| :-------------: | :--------------------------: | :----------: |
| $\alpha_{i, t}$ |  $S_i$ 第一行元素构成的向量  |   $S_i[1]$   |
| $\alpha_{i, b}$ | $S_i$ 最后一行元素构成的向量 |  $S_i[-1]$   |
| $\alpha_{i, l}$ |  $S_i$ 第一列元素构成的向量  | $S_i[:, 1]$  |
| $\alpha_{i, r}$ | $S_i$ 最后一列元素构成的向量 | $S_i[:, -1]$ |

对于相邻的两片纸片 $S_i$ 与 $S_j$，$S_i$ 的右边缘应与 $S_j$ 的左边缘相似，即 $\alpha_{i, r}$ 与 $\alpha_{j, l}$ 相似。参考论文中采用的是比较 $\alpha_{i, r}$ 与 $\alpha_{j, l}$ 中黑色像素点的个数，即比较 $\alpha_{i, r}$ 与 $\alpha_{j, l}$，只考虑了黑色像素点的个数，并没有考虑其位置。因此我们考虑采用余弦相似度来表征 $S_i$ 与 $S_j$ 的匹配度。
$$
m_{ij} = \cos\theta = \dfrac{\left(\alpha_{i, r} \cdot \alpha_{j, l}\right)}{\left\Vert \alpha_{i, r}\right\Vert \left\Vert \alpha_{j, l}\right\Vert}
$$
因为 $S_i$ 中所有元素都为正数，所以 $m_{ij}\in \left[0, 1\right]$。

1. 当 $\alpha_{i, r} = \alpha_{j, l}$ 时，$m_{ij} = 1$，此时 $S_i$ 与 $S_j$ 的匹配度达到最大。
2. 当 $\alpha_{i, r}$ 与 $\alpha_{j, l}$ 正交时，匹配度最小。

### 二维切割的情况

以 $11 \times 19$ 进行切割，得到 $209$ 个纸片，如果直接对所有碎片求距离矩阵，需要计算 $209^2 = 43681$ 次余弦相似度，算法复杂度太高。因此需要对纸片进行聚类达到降维的目的。

#### 聚类

