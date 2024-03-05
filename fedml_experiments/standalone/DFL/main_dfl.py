import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch


# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath("E:\论文\论文1\毕业后\DFL"))
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.model.cv.vgg import vgg16, vgg11
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100
from fedml_api.standalone.DisPFL.dispfl_api import dispflAPI
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.tiny_imagenet.data_loader import load_partition_data_tiny
from fedml_api.model.cv.resnet import  customized_resnet18, tiny_resnet18
#from fedml_api.standalone.DisPFL.my_model_trainer import MyModelTrainer
from fedml_api.standalone.DFL.dfl_api import dflAPI
from fedml_api.standalone.DFL.my_model_trainer import MyModelTrainer

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--model', type=str, default='cnn_cifar100', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar100', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='E:\论文\论文1\毕业后\DFL/',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LRc',
                        help='local learning rate')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay')

    parser.add_argument('--max_norm', type=float, default=10, metavar='max_norm',
                        help='maximum threshold')

    parser.add_argument('--wd', help='weight decay parameter', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='local training epochs for each client')

    parser.add_argument("--topo", type=str, default='random')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=500,
                        help='total communication rounds')

    parser.add_argument('--tau', type=int, default=100,
                        help='topology period')

    parser.add_argument('--frac', type=float, default=0.2, metavar='NN',
                        help='available communication fraction each round')

    parser.add_argument('--momentum', type=float, default=0, metavar='NN',
                        help='momentum')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the test algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--dense_ratio', type=float, default=0.5,
                        help='local density ratio')

    parser.add_argument('--anneal_factor', type=float, default=0.5,
                        help='anneal factor for pruning')

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--cs", type = str, default='random')
    parser.add_argument("--active", type=float,default=1.0)

    parser.add_argument("--public_portion", type=float, default=0)
    parser.add_argument("--erk_power_scale", type=float, default=1 )
    parser.add_argument("--dis_gradient_check", action='store_true')
    parser.add_argument("--strict_avg", action='store_true')
    parser.add_argument("--static", action='store_true')
    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--save_masks", action='store_true')
    parser.add_argument("--different_initial", action='store_true')
    parser.add_argument("--record_mask_diff", action='store_true')
    parser.add_argument("--diff_spa", action='store_true')
    parser.add_argument("--global_test", action='store_true')
    parser.add_argument("--tag", type=str, default="test")
    return parser


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    elif dataset_name == "cifar100":
        args.data_dir += "cifar100"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar100(args.data_dir, args.partition_method,
                                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    elif dataset_name == "tiny":
        args.data_dir += "tiny-imagenet-200"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_tiny(args.data_dir, args.partition_method,
                                             args.partition_alpha, args.client_num_in_total,
                                                 args.batch_size, logger)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset



def create_model(args, model_name,class_num):
    model = None
    if model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != 'tiny':
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    return model


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='CDFL-standalone'))
    args = parser.parse_args()

    print("torch version{}".format(torch.__version__))
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    data_partition=args.partition_method
    if data_partition!="homo":
        data_partition+=str(args.partition_alpha)
    args.identity = "DFL" + "-" + args.dataset + "-" + data_partition
    args.identity+="-mdl" + args.model
    # args.identity+="-cs"+args.cs
    args.identity += "-topo" + args.topo
    args.identity += "-lr" + str(args.lr)
    args.identity += "-epochs" + str(args.epochs)
    args.identity += "-batch_size" + str(args.batch_size)
    args.identity += "-momentum" + str(args.momentum)

    args.identity += "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    args.client_num_per_round = int(args.client_num_in_total* args.frac)
    args.identity += '-tau' + str(args.tau)

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = './LOG/' + args.dataset + '/' + args.identity + '.log'
    if not os.path.exists('./LOG/' + args.dataset):
        os.makedirs('./LOG/' + args.dataset)
    logger = logger_config(log_path=log_path, logging_name=args.identity)


    logger.info(args)
    logger.info(device)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)
    #print(dataset)

    # create model.
    model = create_model(args, model_name=args.model,class_num=len(dataset[-1][0]))
    model_trainer = custom_model_trainer(args, model, logger)
    logger.info(model)
    dflAPI = dflAPI(dataset, device, args, model, model_trainer, logger)
    dflAPI.train()
