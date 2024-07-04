import h5py
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
import xgboost as xgb
import matplotlib.pyplot as plt

# 数据加载和预处理
filename = 'dm3.kc167.example.h5'
try:
    f = h5py.File(filename, 'r')
    x_train = np.array(f['x_train'])
    x_test = np.array(f['x_test'])
    x_val = np.array(f['x_val'])
    y_train = np.array(f['y_train'])
    y_test = np.array(f['y_test'])
    y_val = np.array(f['y_val'])
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 打印数据维度以确认加载正确
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"x_val shape: {x_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"y_val shape: {y_val.shape}")

# Flatten data for classical ML methods
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
x_val_flat = x_val.reshape(x_val.shape[0], -1)

# 打印展平后的数据维度
print(f"x_train_flat shape: {x_train_flat.shape}")
print(f"x_test_flat shape: {x_test_flat.shape}")
print(f"x_val_flat shape: {x_val_flat.shape}")

# # 确保小规模数据集包含足够多的不同类别样本
# def get_balanced_subset(x, y, n_samples):
#     classes = np.unique(y)
#     x_balanced = []
#     y_balanced = []
#     for cls in classes:
#         x_cls = x[y == cls]
#         y_cls = y[y == cls]
#         x_cls, y_cls = shuffle(x_cls, y_cls)
#         x_balanced.append(x_cls[:n_samples // len(classes)])
#         y_balanced.append(y_cls[:n_samples // len(classes)])
#     return np.vstack(x_balanced), np.hstack(y_balanced)

# # 仅使用部分数据进行训练，以缩短训练时间
# x_train_flat_small, y_train_small = get_balanced_subset(x_train_flat, y_train, 5000)

# 定义并评估XGBoost模型
def evaluate_xgboost(x_train, y_train, x_test, y_test):
    try:
        print("Training XGBoost model...")
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr'
        }
        model = xgb.train(params, dtrain, num_boost_round=100)
        print("Model trained. Predicting...")
        y_pred = model.predict(dtest)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)
        return precision, recall, average_precision
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None, None, None

# 定义并评估RandomForest模型
def evaluate_random_forest(x_train, y_train, x_test, y_test):
    try:
        print("Training RandomForest model...")
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        print("Model trained. Predicting...")
        y_pred = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)
        return precision, recall, average_precision
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None, None, None

# 评估XGBoost模型
precision_xgb, recall_xgb, ap_xgb = evaluate_xgboost(x_train_flat, y_train, x_test_flat, y_test)
if precision_xgb is not None:
    print(f"AP (XGBoost): {ap_xgb:.4f}")

# 评估RandomForest模型
precision_rf, recall_rf, ap_rf = evaluate_random_forest(x_train_flat, y_train, x_test_flat, y_test)
if precision_rf is not None:
    print(f"AP (Random Forest): {ap_rf:.4f}")

# 绘制 PR 曲线
plt.figure()
if precision_xgb is not None:
    plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AP={ap_xgb:.2f})')
if precision_rf is not None:
    plt.plot(recall_rf, precision_rf, label=f'Random Forest (AP={ap_rf:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for XGBoost and RandomForest')
plt.legend()
plt.show()
