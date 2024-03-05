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


# # 定义矩阵的大小
# n = 5  # 例如，生成一个5x5的对称矩阵
#
# # 生成一个随机矩阵，元素值为0或1
# random_matrix = np.random.randint(2, size=(n, n))
# np.fill_diagonal(random_matrix, 1)
#
# # 使矩阵对称
# symmetric_matrix = np.triu(random_matrix, k=0) + np.triu(random_matrix, k=1).T
#
# print(symmetric_matrix)
import random
# n_data_per_clnt = [random.randint(200, 4000) for _ in range(5)]
# # 取自然对数
# log_data = np.log(n_data_per_clnt)
#
# # 计算均值和方差
# mean = np.mean(log_data)
# variance = np.var(log_data)
# print(mean)


# # 定义网络节点数n
# client_num_in_total = 20
#
# # 创建一个n*n的全零矩阵
# adjacency_matrix = np.zeros((client_num_in_total, client_num_in_total))
#
# # 设置节点之间的连接，这里假设节点按照二维网格排列
# for i in range(client_num_in_total):
#     for j in range(client_num_in_total):
#         if i > 0:
#             adjacency_matrix[i, (i - 1 + client_num_in_total) % client_num_in_total] = 1  # 连接左侧节点
#         if i < client_num_in_total - 1:
#             adjacency_matrix[i, (i + 1) % client_num_in_total] = 1  # 连接右侧节点
#         if j > 0:
#             adjacency_matrix[i, (i - 5) % client_num_in_total] = 1  # 连接上方节点
#         if j < client_num_in_total - 1:
#             adjacency_matrix[i, (i + 5) % client_num_in_total] = 1  # 连接下方节点
#
# # 输出网格拓扑网络的邻接矩阵
# #print(adjacency_matrix)
# import math
#
# var_topo = random.choices(["ring", "random", "full", "grid"], k=1)
# print(var_topo)
# topos=random.choices(["ring", "random", "full", "grid"], k=1)
# print(math.floor(88/100))
# import torch
# dit={"a":torch.tensor([1.0,1.0,1.0]),"b":torch.tensor([2.0,2.0,2.0]),"c":torch.tensor([3.0,3.0,3.0])}
# total_norm = 0.0
# for key, tensor in dit.items():
#     param_norm = tensor.data.norm(2)
#     total_norm += param_norm.item() ** 2
#     print(param_norm)
#     print(total_norm)
round_m=int(np.floor(5/50))
np.random.seed(round_m)
random_matrix = np.random.randint(2, size=(5, 5))
np.fill_diagonal(random_matrix, 1)
# 使矩阵对称
adjacency_matrix = np.triu(random_matrix, k=0) + np.triu(random_matrix, k=1).T
print(adjacency_matrix)