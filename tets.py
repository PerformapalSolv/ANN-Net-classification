import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd


parser = argparse.ArgumentParser(description='Hand writing digits recognition')
parser.add_argument('-s', '--batch_size', type=int, nargs='?', default=32)
parser.add_argument('-l', '--layer_sizes', nargs='*', type=int, default=[12, 24, 4])
parser.add_argument('-e', '--epoch', nargs='?', default=20, type=int)
parser.add_argument('-q', '--quiet', nargs='?', default=False, type=bool)
parser.add_argument('-p', '--print_score', type=bool, nargs='?', default=True)
parser.add_argument('-o', '--optimizer', type=str, nargs='?', default='minibatch')
parser.add_argument('-f', '--activation_func', type=str, nargs='?', default='sigmoid')
parser.add_argument('-i', '--init_method', type=str, nargs='?', default='uniform')
parser.add_argument('--init_bound', type=float, nargs='*', default=[-0.5, 0.5])
args = parser.parse_args()


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1. - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.square(np.tanh(x))


def softmax(x):
    """
    :param x: 1*n df (n better be small)
    :return: 1*n df
    """
    c = np.max(x)
    tmp = np.exp(x - c)  # 防止溢出
    return tmp / np.sum(tmp)


def one_hot_to_num(y):
    return np.argmax(y)


class Model:
    def __init__(self,
                 layer_sizes,  # 每层的节点数量
                 weight_init_method,  # 初始化weight的方法
                 activation_func_name="ReLU",  # 激活函数，默认为ReLU
                 init_bound=None,
                 input_size=6  # 新增的参数，用于指定输入层的大小
                 ):
        if init_bound is None:
            init_bound = [-0.5, 0.5]
        assert activation_func_name in [
            'sigmoid',
            'tanh'
        ]
        assert weight_init_method in [
            'constant',
            'uniform',
            'normal'
        ]
        if activation_func_name == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_func_derivative = sigmoid_derivative
        elif activation_func_name == 'tanh':
            self.activation_func = tanh
            self.activation_func_derivative = tanh_derivative
        self.init_method_mapper = {
            # arg 此时长度应为1，初始化的目标常数
            'constant': lambda row, column, arg: np.ones(shape=[row, column], dtype='float') * arg[0],
            # arg 此时长度应位2，arg=[lower_bound, upper_bound]
            'uniform': lambda row, column, arg: np.random.rand(row, column) * (arg[1] - arg[0]) + arg[0],
            # arg 此时长度应位2，arg=[mean, variance]
            'normal': lambda row, column, arg: np.random.randn(row, column) * np.sqrt(arg[1]) + arg[0]
        }
        self.layer_num = layer_sizes.__len__()
        self.weight_init_method = self.init_method_mapper[weight_init_method]

        # 将输入层的大小添加到 layer_sizes 的开头
        layer_sizes = [input_size] + layer_sizes

        self.weights = [self.weight_init_method(in_size, out_size, init_bound)
                        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [self.weight_init_method(1, size, init_bound)
                       for size in layer_sizes[1:]]
        self.f1scores_train = []
        self.f1scores_valid = []
        print(f"Model initialization done, with {self.layer_num} layers, layer sizes of {layer_sizes}")

    def forward_backward(self, x, label):
        activations = [x.reshape(1, -1)]
        before_activation = [x.reshape(1, -1)]
        layer = x
        for weight, bias in zip(self.weights, self.biases):
            layer = np.dot(layer, weight) + bias
            before_activation.append(layer)
            layer = self.activation_func(layer)
            activations.append(layer)  # [1,784][1,200][1,100][1,10]
        delta = (softmax(activations[-1]) - label) * self.activation_func_derivative(before_activation[-1])  # [1, 10]
        deriv_b = [np.zeros(b.shape) for b in self.biases]  # [1,200],[1,100],[1,10]
        deriv_w = [np.zeros(w.shape) for w in self.weights]  # [784, 200],[200,100],[100,10]
        deriv_b[-1] = delta.reshape(deriv_b[-1].shape)
        deriv_w[-1] = np.dot(activations[-2].transpose(), deriv_b[-1])
        for i in range(2, self.layer_num):
            delta = np.dot(delta, self.weights[-i + 1].transpose()) * self.activation_func_derivative(
                before_activation[-i])
            deriv_b[-i] = delta
            deriv_w[-i] = np.dot(activations[-i - 1].transpose(), delta)
        return deriv_w, deriv_b

    def minibatch_sgd(self, training_data,
                      test_data,
                      print_score,
                      epoch=2000,
                      minibatch_size=64,
                      step=0.1  # 学习率，默认0.1
                      ):
        training_data = list(training_data)
        test_data = list(test_data)
        data_len = training_data.__len__()
        print('training...')
        for i in tqdm(range(epoch)):
            np.random.shuffle(training_data)
            minibatches = [training_data[j:j + minibatch_size] for j in range(0, data_len, minibatch_size)]
            for data_batch in minibatches:
                self.train_minibatch(data_batch, step)
            if print_score:
                self.f1scores_train.append(np.mean(self.specific_score(training_data)[-1]))
                self.f1scores_valid.append(np.mean(self.specific_score(test_data)[-1]))
        print('done.')
        if print_score:
            print('training data:')
            self.print_score(training_data)
            print('validation data')
            self.print_score(test_data)
        return self.f1scores_train, self.f1scores_valid

    def train_minibatch(self, data, learning_rate):
        total_delta_b = [np.zeros(b.shape) for b in self.biases]
        total_delta_w = [np.zeros(w.shape) for w in self.weights]
        for d in data:
            features = d[:-1]  # 除最后一列外的所有列作为特征
            label = d[-1]  # 最后一列作为标签
            delta_w, delta_b = self.forward_backward(features, label)
            total_delta_b = [t + d for t, d in zip(total_delta_b, delta_b)]
            total_delta_w = [t + d for t, d in zip(total_delta_w, delta_w)]
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, total_delta_w)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, total_delta_b)]

    def standard_sgd(self, train_data, test_data, print_score, epoch=2000, step=0.1):
        train_data = list(train_data)
        test_data = list(test_data)
        print('training...')
        for k in tqdm(range(epoch)):
            np.random.shuffle(train_data)
            for d, l in train_data:
                delta_w, delta_b = self.forward_backward(d, l)
                # 更新参数
                self.weights = [w - step * dw for w, dw in zip(self.weights, delta_w)]
                self.biases = [b - step * db for b, db in zip(self.biases, delta_b)]
            if print_score:
                self.f1scores_train.append(np.mean(self.specific_score(train_data)[-1]))
                self.f1scores_valid.append(np.mean(self.specific_score(test_data)[-1]))
        print('done.')
        if print_score:
            print('training data:')
            self.print_score(train_data)
            print('validation data')
            self.print_score(test_data)
        return self.f1scores_train, self.f1scores_valid

    def get_prediction(self, x):
        t = x
        for bias, weight in zip(self.biases, self.weights):
            t = self.activation_func(np.dot(t, weight) + bias)
        return t

    def print_score(self, data):
        print('=' * 30)
        precise, recall, accuracy, F1 = self.specific_score(data)
        print(f'precision: mean {np.mean(precise)}, variance {np.var(precise)}\n'
              f'recall: mean {np.mean(recall)}, variance {np.var(recall)}')
        print(f'accuracy: mean {np.mean(accuracy)}, variance {np.var(accuracy)}\n'
              f'F1 score: mean {np.mean(F1)}, variance {np.var(F1)}')
        print('=' * 30)

    def specific_score(self, data):
        if isinstance(data[0][-1], np.ndarray):
            data = [(x[:-1], one_hot_to_num(x[-1])) for x in data]
        else:
            data = [(x[:-1], x[-1]) for x in data]
        result = [(np.argmax(self.get_prediction(x)), y) for (x, y) in data]
        return score_mat(result)


def score_mat(predictions):
    predicted_classes = [x for x, _ in predictions]
    true_classes = [y for _, y in predictions]
    num_classes = max(np.max(predicted_classes), np.max(true_classes)) + 1
    mat = np.zeros(shape=(num_classes, num_classes), dtype='int')
    TP = []
    FN = []
    FP = []
    TN = []
    F1 = []
    precise = []
    recall = []
    accuracy = []
    for x, y in predictions:
        if 0 <= x < num_classes and 0 <= y < num_classes:
            mat[x][y] += 1
    for i in range(num_classes):
        TP.append(mat[i][i])
        FN.append(np.sum(mat[:, i]) - mat[i][i])
        FP.append(np.sum(mat[i, :]) - mat[i][i])
        TN.append(len(predictions) - TP[-1] - FN[-1] - FP[-1])
        accuracy.append(float(TP[-1] + TN[-1]) / float(len(predictions)))
        precise.append(float(TP[-1]) / (TP[-1] + FP[-1]) if TP[-1] + FP[-1] != 0 else 0)
        recall.append(float(TP[-1]) / (TP[-1] + FN[-1]) if TP[-1] + FN[-1] != 0 else 0)
        F1.append(2 * precise[-1] * recall[-1] / (precise[-1] + recall[-1]) if precise[-1] + recall[-1] != 0 else 0)
    return precise, recall, accuracy, F1


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

val_df = pd.read_csv('dataset/val.csv', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
# 将特征值转换为数值
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features + ['label']:
    val_df[feature] = val_df[feature].astype('category').cat.codes
X_val = val_df[features].values
y_val = val_df['label'].values
# 对验证集进行标准化
X_val = scaler.transform(X_val)


if __name__ == '__main__':
    print(X_train.shape)
    print(y_train.shape)
    y_train = y_train.reshape(-1, 1)  # 将 y_train 变成二维数组
    y_val = y_val.reshape(-1, 1)  # 将 y_val 变成二维数组
    one_hot_training_data = np.concatenate((X_train, y_train), axis=1)
    validation_data = np.concatenate((X_val, y_val), axis=1)
    test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)  # 将 y_test 变成二维数组
    # draw(train_data[1][0], 28, 28)
    model = Model(layer_sizes=args.layer_sizes, weight_init_method=args.init_method, init_bound=args.init_bound,
                           activation_func_name=args.activation_func)
    print(args)
    assert args.optimizer in ['minibatch', 'standard']
    if args.optimizer == 'minibatch':
        f1_train, f1_valid = model.minibatch_sgd(one_hot_training_data, validation_data,  print_score=bool(args.print_score),
                                                 epoch=args.epoch, minibatch_size=args.batch_size, step=0.1)
    elif args.optimizer == 'standard':
        f1_train, f1_valid = model.standard_sgd(one_hot_training_data, validation_data, print_score=bool(args.print_score),
                                                epoch=args.epoch, step=0.08)
    plt.figure()
    x = range(args.epoch)
    plt.plot(x, f1_train, label='train dataset')
    plt.plot(x, f1_valid, label='validation dataset')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score')
    plt.legend()
    plt.show()
    test_data = list(test_data)
    print('test data:')
    model.print_score(test_data)