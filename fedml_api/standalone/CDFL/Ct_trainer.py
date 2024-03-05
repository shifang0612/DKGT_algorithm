import concurrent.futures
import numpy as np
import torch

class Cti_Trainer:
    def __init__(self, args, device, model, previous_Ct, cur_weight, cur_Zt):
        self.args = args
        self.device = device
        self.model = model
        self.previous_Ct = previous_Ct
        self.cur_weight = cur_weight
        self.cur_Zt = cur_Zt


    def update_Cti_parallel(self, clnt_idx):
        Cti=self.update_Cti(clnt_idx, self.previous_Ct[clnt_idx], self.cur_weight[clnt_idx], self.cur_Zt)
        Cti = self.clip_Cti_norm(Cti)
        return Cti

    def update_Cti(self, cur_idx, previous_Cti, weight, Cur_Zt):
        # previous_Cti 表示上一轮的Cti
        # weight 为一个向量，对应当前client与所有client的整合权重
        Zti = Cur_Zt[cur_idx]
        sum_Zti = {name: sum(
            weight[clnt_id] * Cur_Zt[clnt_id][name].to(self.device) for clnt_id in range(self.args.client_num_in_total))
                   for name, params in self.model.named_parameters()}

        # sum_Sti = {name: sum(
        #     weight[clnt_id] * Cur_St[clnt_id][name].to(self.device) for clnt_id in range(self.args.client_num_in_total))
        #            for name, params in self.model.named_parameters()}

        Cti = {k: previous_Cti[k].to(self.device) + sum_Zti[k] - Cur_Zt[cur_idx][k].to(self.device) for k in Zti.keys()}

        return Cti
    def clip_Cti_norm(self,Cti):
        # Calculate the total gradient norm in Zti
        total_norm = 0.0
        for key, tensor in Cti.items():
            param_norm = tensor.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # Clip the gradients if the total norm exceeds the maximum
        #clip_coef = self.args.max_norm / (total_norm + 1e-6)
        max_norm=1
        clip_coef = max_norm / (total_norm + 1e-6)
        #print(clip_coef)
        if clip_coef < 1.0:
            for key in Cti.keys():
                Cti[key].mul_(clip_coef)  #Zti[key] = Zti[key] * clip_coef
        return Cti

    def train_round_parallel(self):
        # 并行计算 Cur_Ct
        with concurrent.futures.ThreadPoolExecutor() as executor:  # 可以改为ProcessPoolExecutor进行多进程并行
            new_Ct = list(executor.map(self.update_Cti_parallel, range(self.args.client_num_in_total)))

        return new_Ct
