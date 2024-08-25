import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# 数据路径
path = "../data/data_pollution_prepared.xlsx"
data_pollution = pd.read_excel(path, index_col='datetime')

# 选择PM作为预测目标
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

# 训练ARIMA模型
model = ARIMA(train_y, order=(5, 1, 0))  # 基于ARIMA模型
model_fit = model.fit()

# 使用ARIMA模型进行预测
yhat = model_fit.forecast(steps=len(test_y))

# 逆缩放
inv_yhat = np.concatenate((test_X[:, N_STEP * len(data_deal.columns):], yhat.reshape(-1, 1)), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)[:, -1]

inv_y = np.concatenate((test_X[:, N_STEP * len(data_deal.columns):], test_y.reshape(-1, 1)), axis=1)
inv_y = scaler.inverse_transform(inv_y)[:, -1]

# 计算 RMSE、MAE 和 MAPE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100

print(f'Test RMSE: {rmse:.3f}')
print(f'Test MAE: {mae:.3f}')
print(f'Test MAPE: {mape:.3f}%')

# 绘制预测值与真实值对比
plt.figure(figsize=(14, 7))
plt.plot(inv_yhat[:100], label='Predicted PM', color='red')
plt.plot(inv_y[:100], label='Actual PM', color='blue')
plt.legend()
plt.show()

# Test RMSE: 94.752
# Test MAE: 88.515
# Test MAPE: 256.847%