import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from math import sqrt

# 数据路径
path = "../data/data_pollution_prepared.xlsx"
data_pollution = pd.read_excel(path, index_col='datetime')

# 选择特征和标签
data_deal = data_pollution.drop(columns=['PM_label'])  # 去除 PM_label，只保留特征和 PM
labels = data_pollution['PM']  # PM 为预测目标
N_STEP = 1

# 数据预处理
values = data_deal.values
values = values.astype('float32')

# 数据标准化（缩放到0-1之间）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 将时间序列数据转换为监督学习数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 将数据格式化成监督学习型数据
reframed = series_to_supervised(scaled, N_STEP, 1)

# 划分训练集和测试集
values = reframed.values
n_train_hours = 365 * 2 * 24  # 使用两年的数据作为训练集
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 将数据分割成输入和输出
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# 使用SVR模型进行训练
svr_model = SVR(kernel='rbf')
svr_model.fit(train_X, train_y)

# 使用SVR模型进行预测
yhat = svr_model.predict(test_X)

# 在进行逆缩放之前，仅保留预测的PM值
inv_yhat = np.concatenate((test_X[:, N_STEP * len(data_deal.columns) :], yhat.reshape(-1, 1)), axis=1)

# 获取与预测数据对应的原始输入特征
inv_yhat = scaler.inverse_transform(inv_yhat)

# 仅提取逆缩放后的PM值列
inv_yhat = inv_yhat[:, -1]

# 对真实数据进行相同的逆缩放操作
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:, N_STEP * len(data_deal.columns):], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

# 计算 RMSE、MAE 和 MAPE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100

print(f'Test RMSE: {rmse:.3f}')
print(f'Test MAE: {mae:.3f}')
print(f'Test MAPE: {mape:.3f}%')

# 定义标签转换函数
def label_pm(pm):
    if pm <= 35:
        return 1  # '优'
    elif pm <= 75:
        return 2  # '良'
    elif pm <= 115:
        return 3  # '轻度污染'
    elif pm <= 150:
        return 4  # '中度污染'
    elif pm <= 250:
        return 5  # '重度污染'
    else:
        return 6  # '严重污染'

# 将预测的PM值转换为标签
predicted_labels = np.array([label_pm(pm) for pm in inv_yhat])

# 获取实际的PM_label
actual_labels = data_pollution['PM_label'].values[n_train_hours: -N_STEP]  # 对应于测试集的标签

# 计算准确率
accuracy = np.mean(predicted_labels == actual_labels)
print(f'预测的准确率: {accuracy:.4f}')

# 绘制预测值与真实值对比
plt.figure(figsize=(14, 7))
plt.plot(inv_yhat[:100], label='Predicted PM', color='red')
plt.plot(inv_y[:100], label='Actual PM', color='blue')
plt.legend()
plt.show()

subset = 1000  # 或者选择适合你数据的数量
plt.figure(figsize=(14, 7))

for i in range(0, min(len(predicted_labels), subset)):
    if predicted_labels[i] == actual_labels[i]:
        plt.scatter(i, actual_labels[i], color='gray', edgecolor='black', marker='o', alpha=0.6, label='Actual (Correct)' if i == 0 else "")
        plt.scatter(i, predicted_labels[i], color='gray', marker='*', alpha=0.6, label='Predicted (Correct)' if i == 0 else "")
    else:
        plt.scatter(i, actual_labels[i], color='blue', edgecolor='black', marker='o', alpha=0.6, label='Actual (Incorrect)' if i == 0 else "")
        plt.scatter(i, predicted_labels[i], color='red', marker='*', alpha=0.6, label='Predicted (Incorrect)' if i == 0 else "")

plt.title('Comparison of Predicted and Actual Labels')
plt.ylabel('PM Label')
plt.xlabel('Time Step')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# Test RMSE: 25.292
# Test MAE: 22.186
# Test MAPE: 62.329%
# 预测的准确率: 0.4824