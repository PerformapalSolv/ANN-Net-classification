import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义一个基于RNN的神经网络模型
# 定义一个基于RNN的神经网络模型
# 定义一个更深的RNN神经网络模型
class ImprovedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(ImprovedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn1(x, h0)
        out, _ = self.rnn2(out, h0)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, 1)  # 获取预测结果中概率最大的类别
    total = y_true.size(0)  # 样本总数
    correct = (predicted == y_true).sum().item()  # 预测正确的样本数
    return correct / total  # 返回准确率


# 加载数据并进行预处理
df = pd.read_csv('dataset/car_1000.txt', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# 将特征值转换为数值
for feature in features + ['label']:
    df[feature] = df[feature].astype('category').cat.codes

X = df[features].values  # 提取特征值
y = df['label'].values  # 提取标签值

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

# 将数据转换为PyTorch的Tensor格式
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建训练集的DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数、优化器和学习率调度器
model = ImprovedRNN(input_size=6, hidden_size=128, num_layers=3, num_classes=4, dropout_prob=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
# 训练和评估模型
train_accuracies = []  # 存储训练集准确率
val_accuracies = []  # 存储验证集准确率
test_accuracies = []  # 存储测试集准确率
epochs = 200  # 训练轮数
best_val_acc = 0  # 最佳验证集准确率
patience = 20  # early stopping的耐心值
counter = 0  # early stopping的计数器

for epoch in range(epochs):
    model.train()  # 将模型设置为训练模式
    for inputs, labels in train_loader:
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        train_outputs = model(X_train)
        train_accuracy = calculate_accuracy(train_outputs, y_train)  # 计算训练集准确率

        val_outputs = model(X_val)
        val_accuracy = calculate_accuracy(val_outputs, y_val)  # 计算验证集准确率

        test_outputs = model(X_test)
        test_accuracy = calculate_accuracy(test_outputs, y_test)  # 计算测试集准确率

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)

    # 打印每个epoch的准确率
    print(f'Epoch {epoch + 1}, Train Acc: {train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%, Test Acc: {test_accuracy * 100:.2f}%')

    scheduler.step(val_accuracy)  # 根据验证集准确率调整学习率

    # Early stopping机制
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# 绘制准确率随epoch变化的曲线
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Acc')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()
# 保存改进后的RNN模型
model_path = 'improved_rnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Improved RNN model saved to {model_path}')