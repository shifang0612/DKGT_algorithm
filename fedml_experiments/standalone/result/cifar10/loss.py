import ast
import matplotlib.pyplot as plt

log_files=['./DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporandom-batch_size64-lr0.01-max_norm10-epochs5-cm300-total_clnt100-tau50.log',
           './FedAvg-cifar10-dir0.3-mdlcnn_cifar10-toporandom-batch_size64-lr0.01-max_norm10-epochs5-cm300-total_clnt100-tau50.log',
           './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporandom-batch_size64-lr0.01-max_norm10-epochs5-cm300-total_clnt100-tau50.log',
           './DPSGD-cifar10-dir0.3-mdlcnn_cifar10-toporandom-batch_size64-lr0.01-max_norm10-epochs1-cm500-total_clnt100-tau50.log']
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


print(acc_list[0][:100])
print(acc_list[1][:100])
DKGT=acc_list[0]
FedAvg=acc_list[1]
FedAvgM=acc_list[2]
DPSGD=acc_list[3]

#
comm_round=[i for i in range(len(DKGT))]
plt.plot(comm_round, DKGT,label='DKGT')
plt.plot(comm_round, FedAvg,label='DFedAvg')
plt.plot(comm_round, FedAvgM,label='DFedAvgM')
plt.plot(comm_round, DPSGD,label='DPSGD')

plt.title('K=5')
plt.xlabel('communication round')
plt.ylabel('Loss')

plt.legend()

# 显示图表
plt.show()