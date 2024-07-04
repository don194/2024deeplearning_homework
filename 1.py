import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
filename = 'dm3.kc167.example.h5' 
f = h5py.File(filename, 'r') 
x_train = np.array(f['x_train']) 
x_test = np.array(f['x_test']) 
x_val = np.array(f['x_val']) 
y_train = np.array(f['y_train']) 
y_test = np.array(f['y_test']) 
y_val= np.array(f['y_val']) 

# 定义模型
class CNN_LSTM(nn.Module):
    def __init__(self, num_channels, sequence_length, kernel_size, num_kernel, rnn_units):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_kernel, kernel_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(num_kernel, num_kernel, kernel_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 计算池化后的序列长度
        self.lstm_input_size = ((sequence_length - kernel_size + 1) // 2 - kernel_size + 1) // 2
        
        self.lstm = nn.LSTM(num_kernel, rnn_units, batch_first=True, bidirectional=True, dropout=0.5)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * rnn_units * self.lstm_input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.transpose(1, 2)  # 调整输入数据的形状为 (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = x.permute(0, 2, 1)  # 调整形状为 (batch_size, sequence_length, channels) 以适应 LSTM
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        
        return x

# 超参数
INPUT_SHAPE = (4, 1000)  # 独热编码的输入形状（4种碱基 * 1000长度）
KERNEL_SIZE = 9
LEARNING_RATE = 0.001
NUM_KERNEL = 64
RNN_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 40 
outputFile = 'dm3.kc167'

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将 numpy 数据转换为 PyTorch 张量
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建 DataLoader
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNN_LSTM(INPUT_SHAPE[0], INPUT_SHAPE[1], KERNEL_SIZE, NUM_KERNEL, RNN_UNITS).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 测试模型并计算精确率和召回率
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().tolist())
        true_labels.extend(labels.tolist())

# 计算精确率和召回率
precision, recall, _ = precision_recall_curve(true_labels, predictions)
average_precision = average_precision_score(true_labels, predictions)

# 绘制 AUPR 曲线
plt.figure()
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'CNN+RNN Precision-Recall curve')
plt.show()

# 保存模型
torch.save(model.state_dict(), outputFile + '_best_smallCNN_RNN.pth')
