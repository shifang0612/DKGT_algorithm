import ast
import matplotlib.pyplot as plt

# log_files=['./CDFL-cifar10-dir0.3-mdlcnn_cifar10-random-cm50-total_clnt10-K40.log',
#            './CDFL-cifar10-dir0.3-mdlcnn_cifar10-random-cm50-total_clnt10-K60.log',
#            './CDFL-cifar10-dir0.3-mdlcnn_cifar10-random-cm50-total_clnt10-K80.log',
#            './DFL-cifar10-dir0.3-mdlcnn_cifar10-random-cm50-total_clnt10-K40.log',
#            './DFL-cifar10-dir0.3-mdlcnn_cifar10-random-cm50-total_clnt10-K60.log',
#            './DFL-cifar10-dir0.3-mdlcnn_cifar10-random-cm50-total_clnt10-K80.log']
log_files=['./DKGT-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau100.log',
           './FedAvgM-cifar10-dir0.3-mdlcnn_cifar10-toporandom-lr0.01-epochs5-batch_size64-momentum0.9-cm300-total_clnt100-tau100.log']

# log_files=['./CDFL-cifar10-dir0.3-mdlcnn_cifar10-toporandom-cm100-total_clnt20-K20.log',
#            './DFL-cifar10-dir0.3-mdlcnn_cifar10-toporandom-cm100-total_clnt20-K20.log']
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


print(acc_list[0][:100])
print(acc_list[1][:100])
CDFL=acc_list[0]
DFL=acc_list[1]
# acc_CDFL_K40=acc_list[0]
# acc_CDFL_K60=acc_list[1]
# acc_CDFL_K80=acc_list[2]
# acc_DFL_K40=acc_list[3]
# acc_DFL_K60=acc_list[4]
# acc_DFL_K80=acc_list[5]
#
comm_round=[i for i in range(len(CDFL))]
plt.plot(comm_round, CDFL,label='CDFL')
plt.plot(comm_round, DFL,label='DFL')
#
# plt.plot(comm_round, acc_CDFL_K40,label='CDFL_K40')
# plt.plot(comm_round, acc_CDFL_K60,label='CDFL_K60')
# #plt.plot(comm_round, acc_CDFL_K80,label='CDFL_K80')
# plt.plot(comm_round, acc_DFL_K40,label='DFL_K40')
# plt.plot(comm_round, acc_DFL_K60,label='DFL_K60')
#plt.plot(comm_round, acc_DFL_K80,label='DFL_K80')

#plt.title('')
plt.xlabel('communication round')
plt.ylabel('Accuracy')

plt.legend()

# 显示图表
plt.show()