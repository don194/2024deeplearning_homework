import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# 数据加载和预处理
filename = 'dm3.kc167.example.h5'
f = h5py.File(filename, 'r')
x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
x_val = np.array(f['x_val'])
y_train = np.array(f['y_train'])
y_test = np.array(f['y_test'])
y_val = np.array(f['y_val'])

# Convert to PyTorch tensors for neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train_torch = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val_torch = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_torch = torch.tensor(y_val, dtype=torch.float32).to(device)
x_test_torch = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(x_train_torch, y_train_torch)
val_dataset = TensorDataset(x_val_torch, y_val_torch)
test_dataset = TensorDataset(x_test_torch, y_test_torch)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义消融实验模型
class CNN_Only(nn.Module):
    def __init__(self, num_channels, sequence_length, kernel_size, num_kernel):
        super(CNN_Only, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_kernel, kernel_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(num_kernel, num_kernel, kernel_size)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_kernel * ((sequence_length - kernel_size + 1) // 2 - kernel_size + 1) // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

class LSTM_Only(nn.Module):
    def __init__(self, input_dim, sequence_length, rnn_units):
        super(LSTM_Only, self).__init__()
        self.lstm = nn.LSTM(input_dim, rnn_units, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(rnn_units * 2 * sequence_length, 1)  # Correctly compute the input size for the dense layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

# 定义训练和评估函数
def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            true_labels.extend(labels.tolist())

    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    average_precision = average_precision_score(true_labels, predictions)
    return precision, recall, average_precision

def evaluate_classical_ml(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    return precision, recall, average_precision

# 运行实验
# 初始化模型
cnn_only_model = CNN_Only(4, 1000, 9, 64).to(device)
lstm_only_model = LSTM_Only(4, 1000, 64).to(device)  # Update the constructor

# 训练和评估模型
precision_cnn, recall_cnn, ap_cnn = train_and_evaluate(cnn_only_model, train_loader, val_loader, test_loader,30)
precision_lstm, recall_lstm, ap_lstm = train_and_evaluate(lstm_only_model, train_loader, val_loader, test_loader, 20)



# 输出结果
print(f"AP (CNN Only): {ap_cnn:.4f}")
print(f"AP (LSTM Only): {ap_lstm:.4f}")


# 绘制 PR 曲线
plt.figure()
plt.plot(recall_cnn, precision_cnn, label=f'CNN Only (AP={ap_cnn:.2f})')
plt.plot(recall_lstm, precision_lstm, label=f'LSTM Only (AP={ap_lstm:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('CNN Only & LSTM Only Precision-Recall Curve')
plt.legend()
plt.show()
