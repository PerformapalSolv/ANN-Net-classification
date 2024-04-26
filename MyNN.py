import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1. - sigmoid(x))


def softmax(x):
    c = np.max(x)
    tmp = np.exp(x-c)
    return tmp/np.sum(tmp)

class MyNN:
    def __init__(self, layer_sizes, init_bound=None):
        if init_bound is None:
            init_bound = [-0.5, 0.5]
        self.activation_func = sigmoid
        self.activation_func_derivative = sigmoid_derivative
        self.init_method_mapper = {
            # arg 此时长度应位2，arg=[lower_bound, upper_bound]
            'uniform': lambda row, column, arg: np.random.rand(row, column) * (arg[1] - arg[0]) + arg[0],
        }
        self.layer_num = layer_sizes.__len__()
        self.weight_init_method = self.init_method_mapper['uniform']
        self.weights = [self.weight_init_method(in_size, out_size, init_bound)
                        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [self.weight_init_method(1, size, init_bound)
                       for size in layer_sizes[1:]]
        print(f"Model initialization done, with {self.layer_num} layers, layer sizes of {layer_sizes}")

    def forward(self, x):
        t = x
        for weight, bias in zip(self.weights, self.biases):
            t = np.dot(t, weight) + bias
            t = self.activation_func(t)
        return t

    # 实现反向传播算法
    def forward_backward(self, x, label):
        activations = [x]  # 存储每一层的激活值
        zs = []  # 存储每一层的加权输入

        # 前向传播
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activations[-1], weight) + bias
            zs.append(z)
            activation = self.activation_func(z)
            activations.append(activation)
        # 计算输出层的误差
        delta = (activations[-1] - label) * self.activation_func_derivative(zs[-1])
        deriv_b = [np.zeros(b.shape) for b in self.biases]
        deriv_w = [np.zeros(w.shape) for w in self.weights]
        deriv_b[-1] = delta
        deriv_w[-1] = np.dot(activations[-2].reshape(1, -1).T, delta.reshape(1, -1))

        # 反向传播
        for i in range(2, self.layer_num):
            z = zs[-i]
            delta = np.dot(delta, self.weights[-i + 1].T) * self.activation_func_derivative(z)
            deriv_b[-i] = delta
            deriv_w[-i] = np.dot(activations[-i - 1].reshape(1, -1).T, delta.reshape(1, -1))

        return deriv_w, deriv_b

    def standard_sgd(self, train_data, test_data, val_data, epoch=200, step=0.1):
        train_data = list(train_data)
        test_data = list(test_data)
        val_data = list(val_data)

        train_accuracies = []
        test_accuracies = []
        val_accuracies = []
        print('training...')
        for k in tqdm(range(epoch)):
            np.random.shuffle(train_data)
            for x, y in train_data:
                # 将标签转换为 one-hot 编码
                one_hot_label = np.zeros(4)
                one_hot_label[y] = 1
                delta_w, delta_b = self.forward_backward(x, one_hot_label)
                # 更新参数
                self.weights = [w - step * dw for w, dw in zip(self.weights, delta_w)]
                self.biases = [b - step * db for b, db in zip(self.biases, delta_b)]

            # 记录每个epoch的准确率
            train_accuracy = self.accuracy(train_data)
            test_accuracy = self.accuracy(test_data)
            val_accuracy = self.accuracy(val_data)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            val_accuracies.append(val_accuracy)

            # 打印每个epoch的准确率
            print(
                f"Epoch {k + 1}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        return train_accuracies, test_accuracies, val_accuracies
    def predict(self, x):
        return np.argmax(self.forward(x))

    def accuracy(self, data):
        correct = 0
        for x, y in data:
            if self.predict(x) == y:
                correct += 1
        return correct / len(data)


myNN = MyNN(layer_sizes=[6, 12, 24, 4])

# 加载数据并进行预处理
df = pd.read_csv('dataset/car_1000.txt', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features + ['label']:
    df[feature] = df[feature].astype('category').cat.codes
X = df[features].values
y = df['label'].values
# 使用 StandardScaler 对输入数据进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用StratifiedShuffleSplit按照label的比例划分训练集、验证集和测试集
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_val_index], X[test_index]
y_train, y_test = y[train_val_index], y[test_index]
# 加载验证集数据
val_df = pd.read_csv('dataset/val.csv', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])

features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
# 将验证集的特征值转换为数值
for feature in features + ['label']:
    val_df[feature] = val_df[feature].astype('category').cat.codes

X_val = val_df[features].values  # 提取验证集的特征值
y_val = val_df['label'].values  # 提取验证集的标签值

# 对验证集进行标准化
X_val = scaler.transform(X_val)

# 将数据转换为 (x, y) 元组的列表
train_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))
val_data = list(zip(X_val, y_val))


# 训练神经网络并记录准确率
train_accuracies, test_accuracies, val_accuracies = myNN.standard_sgd(train_data, test_data, val_data, epoch=200, step=0.1)

# 评估模型在训练集和测试集上的准确率
train_accuracy = myNN.accuracy(train_data)
test_accuracy = myNN.accuracy(test_data)
val_accuracy = myNN.accuracy(val_data)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")
print(f"Verified accuracy: {val_accuracy:.4f}")

# 可视化准确率变化情况
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Acc')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Acc')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()