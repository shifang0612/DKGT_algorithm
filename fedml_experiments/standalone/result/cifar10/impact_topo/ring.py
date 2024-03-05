import ast
import matplotlib.pyplot as plt


log_files=['./DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporing-batch_size64-lr0.01-max_norm10-epochs3-cm300-total_clnt100-tau300.log',
           './FedAvg-cifar10-dir0.3-mdlcnn_cifar10-toporing-batch_size64-lr0.01-max_norm10-epochs3-cm300-total_clnt100-tau300.log',
           './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporing-batch_size64-lr0.01-max_norm10-epochs3-cm300-total_clnt100-tau300.log',
           './DPSGD-cifar10-dir0.3-mdlcnn_cifar10-toporing-batch_size64-lr0.01-max_norm10-epochs1-cm300-total_clnt100-tau300.log']
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


# print(acc_list[0][:100])
# print(acc_list[1][:100])
# DKGT_K2=acc_list[0]
# DKGT_K3=acc_list[1]
# DKGT_K4=acc_list[2]
# DKGT_K5=acc_list[3]
# DFedAvg_K2=acc_list[4]
# DFedAvg_K3=acc_list[5]
# DFedAvg_K4=acc_list[6]
# DFedAvg_K5=acc_list[7]
DKGT=acc_list[0]
FedAvg=acc_list[1]
FedAvgM=acc_list[2]
DPSGD=acc_list[3]

#
comm_round=[i for i in range(len(DKGT))]
plt.plot(comm_round, DPSGD,label='DPSGD')
plt.plot(comm_round, FedAvg,label='DFedAvg')
plt.plot(comm_round, FedAvgM,label='DFedAvgM')
plt.plot(comm_round, DKGT,label='DKGT')

plt.title('Ring')
plt.xlabel('communication round')
plt.ylabel('Test Accuracy')

plt.legend()
plt.savefig('ring.eps')
# 显示图表
plt.show()