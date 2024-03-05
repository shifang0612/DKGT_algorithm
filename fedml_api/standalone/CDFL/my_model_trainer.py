import copy
import logging
import time

import numpy as np
import pdb
import torch
from torch import nn

from fedml_api.model.cv.cnn_meta import Meta_net

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger = None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def screen_gradients(self, train_data, device):
        model = self.model
        model.to(device)
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels) = next(iter(train_data))
        x, labels = x.to(device), labels.to(device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient={}
        for name, param in model.named_parameters():
            gradient[name] = param.grad.to("cpu")
        return gradient

    # 生成噪声项的函数
    def generate_noise_term(self, device, model_parameters, c_t, magnitude=10):
        noise = {key: torch.randn_like(param) * magnitude for key, param in model_parameters.items()}
        return {key: c_t[key].to(device) * n for key, n in noise.items()}
    # def train(self, train_data, device, args, round, Cti):
    #     # torch.manual_seed(0)
    #     round_lr=args.lr* (args.lr_decay**round)
    #     model = self.model
    #     model.to(device)
    #     model.train()
    #     # train and update
    #     criterion = nn.CrossEntropyLoss().to(device)
    #     if args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=round_lr, momentum=args.momentum,weight_decay=args.wd)
    #     for epoch in range(args.epochs):
    #         epoch_loss = []
    #         for batch_idx, (x, labels) in enumerate(train_data):
    #             x, labels = x.to(device), labels.to(device)
    #             model.zero_grad()
    #             log_probs = model.forward(x)
    #             loss = criterion(log_probs, labels.long())
    #             loss.backward()
    #             # Get the model parameters and gradients
    #             model_parameters = {name: param for name, param in model.named_parameters()}
    #             gradients = {name: param.grad for name, param in model_parameters.items()}
    #             # # 生成带有 C_t 乘法的噪声项
    #             # noise = self.generate_noise_term(device,model_parameters,Cti)
    #             for name, param, grad, Ci in zip(model_parameters.keys(), model_parameters.values(), gradients.values(),
    #                                             Cti.values()):
    #                 if grad is not None:
    #                     param.data -= round_lr * (grad + Ci.to(device))
    #                     #print("grad + Ci.to(device):{}".format(grad + Ci.to(device)))
    #
    #             # Optional: Clip gradients to avoid exploding gradients
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
    #             optimizer.step()
    #             epoch_loss.append(loss.item())
    #
    #         self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #             self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
    #     return round_lr
    def train(self, train_data,  device,  args, round, Cti):
        # torch.manual_seed(0)
        round_lr=args.lr* (args.lr_decay**round)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        #grad_sum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        #K=int(len(train_data)/self.args.batch_size)
        criterion = nn.CrossEntropyLoss().to(device)
        # 添加 L2 正则化
        l2_lambda = 0.001  # 超参数，控制正则化的强度
        # reg_term = 0.0
        # for name, param in model.named_parameters():
        #     reg_term += l2_lambda * torch.norm(param, p=2)
        torch.autograd.set_detect_anomaly(True)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=round_lr, momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                log_probs = model.forward(x)
                #loss = criterion(log_probs, labels.long()) + l2_lambda * sum(torch.norm(param, p=2) for param in model.parameters())
                loss = criterion(log_probs, labels.long())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                # optimizer.step()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.data -= round_lr * (param.grad + Cti[name].to(device))
                        #grad_sum[name] += (param.grad + Cti[name].to(device)) / self.args.epochs
                        #param.data -= round_lr * Cti[name].to(device)
                        # new_param = param - round_lr * (param.grad + Cti[name].to(device))
                        # param.data.copy_(new_param)
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
                #optimizer.zero_grad()
                epoch_loss.append(loss.item())
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        return round_lr
    # def train(self, train_data,  device,  args, round, Cti):
    #     # torch.manual_seed(0)
    #     round_lr=args.lr* (args.lr_decay**round)
    #     model = self.model
    #     model.to(device)
    #     model.train()
    #     # train and update
    #     #grad_sum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    #     #K=int(len(train_data)/self.args.batch_size)
    #     criterion = nn.CrossEntropyLoss().to(device)
    #     # 添加 L2 正则化
    #     l2_lambda = 0.001  # 超参数，控制正则化的强度
    #     # reg_term = 0.0
    #     # for name, param in model.named_parameters():
    #     #     reg_term += l2_lambda * torch.norm(param, p=2)
    #     torch.autograd.set_detect_anomaly(True)
    #     if args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=round_lr, momentum=args.momentum,weight_decay=args.wd)
    #     for epoch in range(20):
    #         epoch_loss = []
    #         (x, labels) = next(iter(train_data))
    #         x, labels = x.to(device), labels.to(device)
    #         log_probs = model.forward(x)
    #         # loss = criterion(log_probs, labels.long()) + l2_lambda * sum(torch.norm(param, p=2) for param in model.parameters())
    #         loss = criterion(log_probs, labels.long())
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         # to avoid nan loss
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    #         optimizer.step()
    #         for name, param in model.named_parameters():
    #             if param.grad is not None:
    #                 # grad_sum[name] += (param.grad + Cti[name].to(device)) / self.args.epochs
    #                 param.data -= round_lr * Cti[name].to(device)
    #                 # new_param = param - round_lr * (param.grad + Cti[name].to(device))
    #                 # param.data.copy_(new_param)
    #         # to avoid nan loss
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    #         optimizer.step()
    #         # optimizer.zero_grad()
    #         epoch_loss.append(loss.item())
    #         self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #             self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
    #
    #
    #     return round_lr

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

