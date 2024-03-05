import ast
import matplotlib.pyplot as plt

log_file = './LOG/cifar10/DFL-cifar10-dir0.3-mdlcnn_cifar10-toporandom-batch_size128-cm100-total_clnt100-tau50-K15.log'  # 日志文件名
#search_text = 'test_acc'  # 要查找的特定文本
#search_text = 'test_loss'
search_text = 'gradient of global model'
acc_list=[]
test_loss=[]
grad_norm=[]
with open(log_file, 'r', encoding='utf-8') as file:
    for line in file:
        if search_text in line:
            log_data = ast.literal_eval(line)
            #print(line)
            test_acc_value = log_data['test_acc']
            #print(test_acc_value)
            acc_list.append(test_acc_value)
            test_loss.append(log_data['test_loss'])
            grad_norm.append(log_data['gradient of global model'])

print(acc_list)
comm_round=[i for i in range(len(acc_list))]

plt.plot(comm_round, grad_norm)

#plt.title('')
plt.xlabel('communication round')
plt.ylabel('test loss')

# 显示图表
plt.show()