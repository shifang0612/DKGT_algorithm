import torch.nn as nn
import numpy as np
#
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3)
#         self.fc1 = nn.Linear(64, 10)
#
# model = MyModel()
#
# # 使用named_parameters迭代参数
# for name, param in model.named_parameters():
#     print(f"Parameter name: {param}")
#
# # 使用items迭代子模块和属性
# for name, module in model.named_children():
#     print(f"Child module name: {param}")

# # 定义节点数
# num_nodes = 6  # 例如，6个节点
#
# # 创建一个邻接矩阵表示圆环拓扑网络
# adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
#
# # 设置连接
# for i in range(num_nodes):
#     adjacency_matrix[i, (i + 1) % num_nodes] = 1  # 每个节点与下一个节点相连
#
# # 打印邻接矩阵
# print(adjacency_matrix)
#
# # 创建一个邻接列表表示圆环拓扑网络
# adjacency_list = {}
#
# # 创建邻接列表
# for i in range(num_nodes):
#     neighbors = [(i + 1) % num_nodes]  # 每个节点的邻居是下一个节点
#     adjacency_list[i] = neighbors
#
# # 打印邻接列表
# for node, neighbors in adjacency_list.items():
#     print(f"Node {node}: Neighbors {neighbors}")
#
# for i in range(5):
#     print(i)


# 定义矩阵的大小
n = 5  # 例如，生成一个5x5的对称矩阵

# 生成一个随机矩阵，元素值为0或1
random_matrix = np.random.randint(2, size=(n, n))
np.fill_diagonal(random_matrix, 1)

# 使矩阵对称
symmetric_matrix = np.triu(random_matrix, k=0) + np.triu(random_matrix, k=1).T

print(symmetric_matrix)

