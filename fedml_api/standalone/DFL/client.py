import copy
import logging
import math

import numpy as np
import pdb
import torch

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def client_gradient(self, w):
        self.model_trainer.set_model_params(w)
        self.model_trainer.set_id(self.client_idx)
        gradient = self.model_trainer.screen_gradients(self.local_training_data, self.device)
        return gradient


    def train(self, w, round):
        # downlink params
        #num_comm_params = self.model_trainer.count_communication_params(w)
        self.model_trainer.set_model_params(w)
        # self.model_trainer.set_masks(masks)
        self.model_trainer.set_id(self.client_idx)
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))

        self.model_trainer.train(self.local_training_data, self.device, self.args, round)
        parameters = self.model_trainer.get_model_params()
        self.model_trainer.set_model_params(parameters)
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        self.logger.info("test acc on this client after {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        gradient = self.model_trainer.screen_gradients(self.local_training_data, self.device)

        # full_flops = self.model_trainer.count_full_flops_per_sample()
        # training_flops = self.args.epochs * self.args.batch_size * full_flops

        # uplink paramss
        #num_comm_params += self.model_trainer.count_communication_params(parameters)
        return parameters, tst_results, gradient


    # def fire_mask(self, masks, weights, round):
    #     drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / self.args.comm_round))
    #     new_masks = copy.deepcopy(masks)
    #
    #     num_remove = {}
    #     for name in masks:
    #         num_non_zeros = torch.sum(masks[name])
    #         num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
    #         temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]), 100000 * torch.ones_like(weights[name]))
    #         x, idx = torch.sort(temp_weights.view(-1).to(self.device))
    #         new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
    #     return new_masks, num_remove

    #
    # # we only update the private components of client's mask
    # def regrow_mask(self, masks,  num_remove, gradient=None):
    #     new_masks = copy.deepcopy(masks)
    #     for name in masks:
    #         # if name not in public_layers:
    #             # if "conv" in name:
    #             if not self.args.dis_gradient_check:
    #                 temp = torch.where(masks[name] == 0, torch.abs(gradient[name]), -100000 * torch.ones_like(gradient[name]))
    #                 sort_temp, idx = torch.sort(temp.view(-1).to(self.device), descending=True)
    #                 new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
    #             else:
    #                 temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
    #                 idx = torch.multinomial( temp.flatten().to(self.device),num_remove[name], replacement=False)
    #                 new_masks[name].view(-1)[idx]=1
    #     return new_masks


    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
