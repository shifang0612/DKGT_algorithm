import ast
import matplotlib.pyplot as plt

log_file = './LOG/cifar10/CDFL-cifar10-dir0.3-mdlcnn_cifar10-DST-cm100-total_clnt10-dr0.5.log'  # 日志文件名
search_text = 'test_acc'  # 要查找的特定文本
acc_list=[]
with open(log_file, 'r', encoding='utf-8') as file:
    for line in file:
        if search_text in line:
            log_data = ast.literal_eval(line)
            #print(line)
            test_acc_value = log_data['test_acc']
            #print(test_acc_value)
            acc_list.append(test_acc_value)

print(acc_list)
comm_round=[i for i in range(100)]

plt.plot(comm_round, acc_list)

#plt.title('')
plt.xlabel('communication round')
plt.ylabel('Accuracy')

# 显示图表
plt.show()