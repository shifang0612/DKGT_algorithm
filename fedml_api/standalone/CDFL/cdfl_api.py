import copy
import logging
import math
import pickle
import random
import time


import pdb
import numpy as np
import torch

from fedml_api.standalone.CDFL import client
from fedml_api.standalone.CDFL.client import Client
from fedml_api.standalone.CDFL.St_trainer import Sti_Trainer
from fedml_api.standalone.CDFL.Ct_trainer import Cti_Trainer
from collections import defaultdict

class cdflAPI(object):
    def __init__(self, dataset, device, args, model, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_counts = class_counts
        self.model=model
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
        self.topo=args.topo

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")


    def train(self):
        # initialize the local model of each node as zero
        w_global = self.model_trainer.get_model_params()
        Cti_ini = {}  # record the parameter of gradient tracking
        ini_gradient_sum = {}
        zero_model={}
        for name, param in self.model.named_parameters():
            Cti_ini[name] = torch.zeros_like(param)
            ini_gradient_sum[name]=torch.zeros_like(param)
            zero_model[name] = torch.zeros_like(param)

        w_local_mdls = []
        Ct_previous=[]
        #St_previous=[]
        Zt_previous=[]
        # 初始化
        for clnt in range(self.args.client_num_in_total):
            w_local_mdls.append(copy.deepcopy(w_global))
            Ct_previous.append(copy.deepcopy(Cti_ini))
            # St_previous.append(copy.deepcopy(Cti_ini))
            Zt_previous.append(copy.deepcopy(Cti_ini))
        # pre_weight_matrix=np.zeros((self.args.client_num_in_total,self.args.client_num_in_total))
        random.seed(1)
        topos=random.choices(["ring", "random", "full", "grid"], k=int(self.args.comm_round/self.args.tau))
        #topos = ["ring", "random", "full", "grid"]
        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            round_m=int(math.floor(round_idx/self.args.tau))
            var_topo=topos[round_m]
            #var_topo=self.args.topo
            self.logger.info('topology:{}'.format(var_topo))
            #确定topology矩阵和权重矩阵
            topo_matrix=self.round_topology(round_m,self.args.client_num_in_total, var_topo)
            cur_weight_matrix, neighbor_matrix = self.obtain_weight_matrix(topo_matrix,self.args.client_num_in_total)
            #print("topo matrix:{}, weight matrix:{}, neighbor matrix:{}".format(topo_matrix, weight_matrix, neighbor_matrix))

            # 更新communication round时的所有local model
            w_local_mdls_lstrd = copy.deepcopy(w_local_mdls)

            # 在每一个communication rounds需要进行每个client的local training
            tst_results_ths_round = [] #保存每个client本地训练完的结果
            Cur_Zt=[] # record Zti of all nodes,一行对应一个节点的Zti
            local_grads=[]
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}) {}'.format(round_idx, clnt_idx))
                # Update each client's local model and the so-called consensus model
                w_local_mdl = self._aggregate_func(clnt_idx, neighbor_matrix[clnt_idx], cur_weight_matrix, w_local_mdls_lstrd)

                # 设置client进行local training
                client = self.client_list[clnt_idx]
                round_lr, w_trained_mdl, tst_results, gradient = client.train(w_local_mdl, round_idx, Ct_previous[clnt_idx])
                #print("w_trained_mdl:{}".format(w_trained_mdl))
                tst_results_ths_round.append(tst_results)
                local_grads.append(gradient)
                logging.info("{} comm, client {}, test result after train: {}".format(round_idx, clnt_idx,
                                                                                           tst_results))
                # # update Zti
                Zti=self.update_Zti(w_local_mdl, w_trained_mdl, round_lr)
                Cur_Zt.append(Zti)
                # 更新local model
                w_local_mdls[clnt_idx] = copy.deepcopy(w_trained_mdl)
            # calculate the norm of average
            ave_grad_norm=self.grad_norm(local_grads)
            self.logger.info("ave_grad_norm:{}".format(ave_grad_norm))
            # calculate the consensus distance
            cons_dis = self.cons_dis(w_local_mdls)
            self.logger.info("consensus_distance:{}".format(cons_dis))
            # update Sti
            # st_trainer=Sti_Trainer(self.args, self.device, self.model, St_previous,  pre_weight_matrix, Zt_previous, cur_weight_matrix, Cur_Zt)
            # Cur_St=st_trainer.train_round_parallel()
            ct_trainer=Cti_Trainer(self.args, self.device, self.model, Ct_previous, cur_weight_matrix, Cur_Zt)
            Cur_Ct=ct_trainer.train_round_parallel()
            Ct_previous=Cur_Ct

            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
        return

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    def round_topology(self,round_m,client_num_in_total,topo):
        if topo=="ring":
            # 创建一个邻接矩阵表示圆环拓扑网络
            adjacency_matrix = np.zeros((client_num_in_total, client_num_in_total), dtype=int)
            # 设置连接
            for i in range(client_num_in_total):
                adjacency_matrix[i, (i - 1 + client_num_in_total) % client_num_in_total] = 1  # 设置左连接
                adjacency_matrix[i, (i + 1) % client_num_in_total] = 1  # 设置右连接
        elif topo=="full":
            adjacency_matrix = np.ones((client_num_in_total, client_num_in_total), dtype=int)
        elif topo=="random":
            # 生成一个随机矩阵，元素值为0或1
            np.random.seed(round_m)
            random_matrix = np.random.randint(2, size=(client_num_in_total, client_num_in_total))
            np.fill_diagonal(random_matrix, 1)
            # 使矩阵对称
            adjacency_matrix = np.triu(random_matrix, k=0) + np.triu(random_matrix, k=1).T
            m=20 #20为接入的数量
            # 确保每行为1的数的个数为m
            for i in range(client_num_in_total):
                row_sum = np.sum(adjacency_matrix[i, :])
                if row_sum > m:
                    # 随机选择超出m的元素置零
                    np.random.seed(round_m)
                    indices_to_zero = np.random.choice(np.where(adjacency_matrix[i, :] == 1)[0], size=row_sum - m,
                                                       replace=False)
                    adjacency_matrix[i, indices_to_zero] = 0
                elif row_sum < m:
                    # 随机选择不足m的元素置一
                    np.random.seed(round_m)
                    indices_to_one = np.random.choice(np.where(adjacency_matrix[i, :] == 0)[0], size=m - row_sum,
                                                      replace=False)
                    adjacency_matrix[i, indices_to_one] = 1
        elif topo=="grid":
            adjacency_matrix = np.zeros((client_num_in_total, client_num_in_total))
            for i in range(client_num_in_total):
                for j in range(client_num_in_total):
                    if i > 0:
                        adjacency_matrix[i, (i - 1 + client_num_in_total) % client_num_in_total] = 1  # 连接左侧节点
                    if i < client_num_in_total - 1:
                        adjacency_matrix[i, (i + 1) % client_num_in_total] = 1  # 连接右侧节点
                    if j > 0:
                        adjacency_matrix[i, (i - 5) % client_num_in_total] = 1  # 连接上方节点
                    if j < client_num_in_total - 1:
                        adjacency_matrix[i, (i + 5) % client_num_in_total] = 1  # 连接下方节点
        return adjacency_matrix

    def _aggregate_func(self, cur_idx, nei_indexs, agg_weight, w_mdls):
        #clnt_idx, nei_indexs_matrix[clnt_idx], agg_weight_matrix[clnt_idx],
        #w_per_mdls_lstrd, Zti_per_matrix
        self.logger.info('Doing local aggregation!')
        # Use the received models to infer the consensus model
        w_tmp = copy.deepcopy(w_mdls[cur_idx])
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in nei_indexs:
                w_tmp[k] += agg_weight[cur_idx][clnt] * w_mdls[clnt][k]
        return w_tmp

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
            self.logger.info("################local_test_on_all_clients after local training in communication round: {}".format(round_idx))
            test_metrics = {
                'num_samples': [],
                'num_correct': [],
                'losses': []
            }
            for client_idx in range(self.args.client_num_in_total):
                # test data
                test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
                test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
                test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # # test on test dataset
            test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in range(self.args.client_num_in_total) ] )/self.args.client_num_in_total
            test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in range(self.args.client_num_in_total)])/self.args.client_num_in_total

            stats = {'test_acc': test_acc, 'test_loss': test_loss}

            self.logger.info(stats)


    def init_stat_info(self, ):
        self.stat_info = {}
        self.stat_info["label_num"] =self.class_counts
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []

    def cons_dis(self, local_models):
        #global_model = defaultdict(float)
        global_model = {}
        for key, value in local_models[0].items():
            global_model[key] = torch.zeros_like(value)
        for local_model in local_models:
            for key, value in local_model.items():
                global_model[key] += value/len(local_models)

        consensus_distance = 0.0
        for local_model in local_models:
            model_distance = 0.0
            for local_param, global_param in zip(local_model.values(), global_model.values()):
                param_distance = torch.norm(local_param - global_param, p=2)
                model_distance += param_distance.item()
            consensus_distance += model_distance
        average_consensus_distance = consensus_distance / len(local_models)

        return average_consensus_distance

    def comp_norm(self, model_dist):
        total_norm = 0.0
        for key, tensor in model_dist.items():
            param_norm = tensor.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm

    def obtain_weight_matrix(self, topo_matrix, client_num_in_total):
        weight_matrix=np.zeros((client_num_in_total,client_num_in_total))
        neighbor_matrix=[]
        for i in range(client_num_in_total):
            neis=[]
            for j in range(client_num_in_total):
                if i!=j and topo_matrix[i][j]>0:
                    neis.append(j)
                    weight_matrix[i][j] =1/(1+max(sum(topo_matrix[i]),sum(topo_matrix[j])))
            neis.append(i)
            neis=np.sort(neis)
            weight_matrix[i][i] = 1 - sum(weight_matrix[i])
            neighbor_matrix.append(neis)
        return weight_matrix, neighbor_matrix

    def update_Zti(self, w_local_mdl, w_trained_mdl, round_lr):
        Zti = {key: (w_local_mdl[key] - w_trained_mdl[key]) / (self.args.epochs * round_lr) for key in w_trained_mdl.keys()}
        return Zti

    def grad_norm(self,local_gradients):
        # 确保有梯度可聚合
        if len(local_gradients) == 0:
            return None
        # 初始化全局梯度字典
        global_gradient = {}
        # 对每个参数的梯度进行分别的平均
        for param_name in local_gradients[0].keys():
            # 使用 PyTorch 的 torch.stack 将梯度堆叠在一起
            stacked_gradients = torch.stack([grad[param_name] for grad in local_gradients])

            # 对堆叠的梯度取平均，得到全局梯度
            global_gradient[param_name] = torch.mean(stacked_gradients, dim=0)
        grad_norm=0.0
        for value in global_gradient.values():
            norm=torch.norm(value,p=2)
            grad_norm += norm
        # # # 计算全局梯度范数
        # global_gradient_norm = torch.norm(torch.stack([global_gradient[param_name] for param_name in global_gradient.values()]))

        return grad_norm/len(local_gradients)





