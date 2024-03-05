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
search_text = 'test_acc'  # 要查找的特定文本
acc_list=[]
for file_name in log_files:
    list=[]
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if search_text in line:
                log_data = ast.literal_eval(line)
                # print(line)
                test_acc_value = log_data['test_acc']
                # print(test_acc_value)
                list.append(test_acc_value)
    acc_list.append(list)


DKGT=acc_list[0]
FedAvg=acc_list[1]
FedAvgM=acc_list[2]
DPSGD=acc_list[3]


comm_round=[i for i in range(len(DKGT))]
plt.plot(comm_round, DPSGD,label='DPSGD')
plt.plot(comm_round, FedAvg,label='DFedAvg')
plt.plot(comm_round, FedAvgM,label='DFedAvgM')
plt.plot(comm_round, DKGT,label='DKGT')
plt.title(r'$\tau=T$')
plt.xlabel('communication round')
plt.ylabel('Test Accuracy')

plt.legend()

plt.savefig('tau300_r.eps')
# 显示图表
plt.show()