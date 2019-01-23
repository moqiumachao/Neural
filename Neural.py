import numpy as np

#生成训练样本
x_data = np.float32(np.random.rand(2,400))
y_data = np.dot([0.700,0.200],x_data)+4

#激活函数
def activation(x):
    if x > 0:
        return x
    else:
        return 0

#创建神经元类
class NeuralNet(object):
    def __init__(self):
        self.weights = np.float64([[0.1,0.2]])
        self.bias = np.float64(0)
        self.learn_rate = 1
    def train(self,input_data,output_data):
        net = np.dot(self.weights,input_data)+self.bias
        y = activation(net)
        dw = self.learn_rate * (output_data-y)*input_data
        self.weights = self.weights + dw#更新权重
        db = self.learn_rate * (output_data-y)
        self.bias = self.bias + db#更新偏置
    def show_weights(self):
        print("weights:",self.weights,end=' ')
    def show_bias(self):
        print("bias:",self.bias)
        
#创建单个神经元对象
neural = NeuralNet()

#train
for i in range(400):
    neural.train(x_data[:,i],y_data[i])
    if i%50==0:
        print('the %d step is:'%i)
        neural.show_weights()
        neural.show_bias()   
        
