import concurrent.futures
import numpy as np
import torch

class Sti_Trainer:
    def __init__(self, args, device, model, previous_Sti, previous_weight, previous_Zt, cur_weight, cur_Zt):
        self.args = args
        self.device = device
        self.model = model
        self.previous_Sti = previous_Sti
        self.previous_weight = previous_weight
        self.previous_Zt = previous_Zt
        self.cur_weight = cur_weight
        self.cur_Zt = cur_Zt

    def update_Sti_parallel(self, clnt_idx):
        return self.update_Sti(self.previous_Sti[clnt_idx], self.previous_weight[clnt_idx], self.previous_Zt,
                               self.cur_weight[clnt_idx], self.cur_Zt)

    def update_Sti(self, previous_Sti, previous_weight, previous_Zt, cur_weight, cur_Zt):
        # weight 为一个向量，对应当前client与所有client的整合权重
        # St_previous[clnt_idx], weight_pre_matrix[clnt_idx], Zt_previous, weight_matrix[clnt_idx], Cur_Zt
        sum_pre_Zti = {
            name: sum(previous_weight[clnt_id] * previous_Zt[clnt_id][name].to(self.device) for clnt_id in
                      range(self.args.client_num_in_total)) for name, params in self.model.named_parameters()}

        sum_cur_Cti = {name: sum(cur_weight[clnt_id] * cur_Zt[clnt_id][name].to(self.device) for clnt_id in
                                 range(self.args.client_num_in_total)) for name, params in
                       self.model.named_parameters()}

        Sti = {k: previous_Sti[k].to(self.device) - sum_pre_Zti[k] + sum_cur_Cti[k] for k in sum_cur_Cti.keys()}

        return Sti

    def train_round_parallel(self):
        # 并行计算 Cur_St
        with concurrent.futures.ThreadPoolExecutor() as executor:
            Cur_St = list(executor.map(self.update_Sti_parallel, range(self.args.client_num_in_total)))

        return Cur_St
