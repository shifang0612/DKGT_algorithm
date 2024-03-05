import ast
import matplotlib.pyplot as plt

# log_files=['./DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau1.log',
#            './DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau100.log',
#            './DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau300.log',
#            './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau1.log',
#            './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau100.log',
#            './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau300.log']

log_files=['./DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau300.log',
           './FedAvg-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0-cm300-total_clnt100-tau300.log',
           './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau300.log',
           './DPSGD-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs1-batch_size64-momentum0-cm300-total_clnt100-tau300.log']
search_text = 'test_loss'  # 要查找的特定文本
acc_list=[]
for file_name in log_files:
    list=[]
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if search_text in line:
                log_data = ast.literal_eval(line)
                # print(line)
                test_acc_value = log_data['test_loss']
                # print(test_acc_value)
                list.append(test_acc_value)
    acc_list.append(list)


# print(acc_list[0][:100])
# print(acc_list[1][:100])
# DKGT_tau1=acc_list[0]
# DKGT_tau100=acc_list[1]
# DKGT_tau300=acc_list[2]
# FedAvg_tau1=acc_list[3]
# FedAvg_tau100=acc_list[4]
# FedAvg_tau300=acc_list[5]
#
# #
# comm_round=[i for i in range(len(DKGT_tau1))]
# plt.plot(comm_round, DKGT_tau1,label='DKGT_tau1')
# plt.plot(comm_round, DKGT_tau100,label='DKGT_tau100')
# plt.plot(comm_round, DKGT_tau300,label='DKGT_tau300')
# plt.plot(comm_round, FedAvg_tau1,label='FedAvg_tau1')
# plt.plot(comm_round, FedAvg_tau100,label='FedAvg_tau100')
# plt.plot(comm_round, FedAvg_tau300,label='FedAvg_tau300')

DKGT=acc_list[0]
FedAvg=acc_list[1]
FedAvgM=acc_list[2]
DPSGD=acc_list[3]


comm_round=[i for i in range(len(DKGT))]
plt.plot(comm_round, DKGT,label='DKGT')
plt.plot(comm_round, FedAvg,label='DFedAvg')
plt.plot(comm_round, FedAvgM,label='DFedAvgM')
plt.plot(comm_round, DPSGD,label='DPSGD')
plt.title('$tau=300$')
plt.xlabel('communication round')
plt.ylabel('Accuracy')

plt.legend()

# 显示图表
plt.show()