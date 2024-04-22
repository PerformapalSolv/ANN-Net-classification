import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return correct / total

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

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

model = SimpleNN(input_size=6, hidden_size=10, num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_accuracies = []
test_accuracies = []
val_accuracies = []
epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        train_accuracy = calculate_accuracy(train_outputs, y_train)

        val_outputs = model(X_val)
        val_accuracy = calculate_accuracy(val_outputs, y_val)

        test_outputs = model(X_test)
        test_accuracy = calculate_accuracy(test_outputs, y_test)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch + 1}, Train Acc: {train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%, Test Acc: {test_accuracy * 100:.2f}%')

    # scheduler.step(val_accuracy)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Acc')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

# 指定模型保存的路径
model_path = 'trained_model.pth'

# 保存模型的状态字典
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')