import pandas as pd
import numpy as np
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, concatenate, Reshape, RepeatVector
from keras.callbacks import EarlyStopping
from math import sqrt

# 数据路径
path = "../data/data_pollution_prepared.xlsx"
data_pollution = pd.read_excel(path, index_col='datetime')

# # 定义要去除的日期范围
# start_date = '2013-05-04'
# end_date = '2013-05-23'
#
# # 去掉指定日期范围的数据
# data_pollution = data_pollution.drop(data_pollution.loc[start_date:end_date].index)
# 选择特征和标签
data_deal = data_pollution.drop(columns=['PM_label'])  # 去除 PM_label，只保留特征和 PM
labels = data_pollution['PM']  # PM 为预测目标
N_DATA = 13
N_STEP = 1
N_EPOCHS = 10000

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

# 将输入数据转换成3D张量 [样本数, 时间步长, 特征数]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# 构建Inception-like块
def inception_block(input_layer):
    # LSTM分支
    lstm_branch = LSTM(30)(input_layer)

    # GRU分支
    gru_branch = GRU(30)(input_layer)

    # LSTM + GRU分支
    lstm_gru_branch = LSTM(20, return_sequences=True)(input_layer)
    lstm_gru_branch = GRU(20)(lstm_gru_branch)

    # 合并分支
    merged = concatenate([lstm_branch, gru_branch, lstm_gru_branch])

    return merged

# 输入层
input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))

# 第一层 Inception-like 块
x = inception_block(input_layer)

# 调整维度，使其兼容下一层 LSTM/GRU 层
x = RepeatVector(1)(x)  # 增加一个时间步
x = Reshape((1, -1))(x)  # 将数据调整为 3D 张量

# 第二层 Inception-like 块
x = inception_block(x)

# 输出层
output_layer = Dense(1)(x)
custom_lr = 0.0002  # 自定义学习率
# 创建模型
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mae', optimizer=Adam(learning_rate=custom_lr))

# 定义早停回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

# 训练模型
history = model.fit(train_X, train_y, epochs=N_EPOCHS, batch_size=24, validation_data=(test_X, test_y),
                    verbose=2, shuffle=False, callbacks=[early_stopping])

# 绘制训练损失和验证损失
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 使用模型进行预测
yhat = model.predict(test_X)

# 将预测数据的形状调整为与原始数据匹配
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# 在进行逆缩放之前，仅保留预测的PM值
inv_yhat = np.concatenate((test_X[:, N_STEP * N_DATA:], yhat), axis=1)

# 获取与预测数据对应的原始输入特征
inv_yhat = scaler.inverse_transform(inv_yhat)

# 仅提取逆缩放后的PM值列
inv_yhat = inv_yhat[:, -1]

# 对真实数据进行相同的逆缩放操作
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:, N_STEP * N_DATA:], test_y), axis=1)
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

# n_step = 1, lr=0.001, batch_size=72
# Test RMSE: 9.383
# Test MAE: 6.620
# Test MAPE: 13.352%
# 预测的准确率: 0.9066

# Test RMSE: 8.818
# Test MAE: 6.246
# Test MAPE: 12.638%
# 预测的准确率: 0.9208

# n_step = 1, lr=0.001, batch_size=48
# Test RMSE: 8.499
# Test MAE: 5.879
# Test MAPE: 11.477%
# 预测的准确率: 0.9346

# n_step = 1, lr=0.001, batch_size=24
# Test RMSE: 8.261
# Test MAE: 5.621
# Test MAPE: 11.224%
# 预测的准确率: 0.9444