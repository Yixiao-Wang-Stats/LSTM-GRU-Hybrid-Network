import pandas as pd
from DataPrepareFunctions import *


##############################################
#                   文件导入
##############################################
path = "../data/ChengduPM20100101_20151231.csv"
data_pollution = pd.read_csv(path)

##############################################
#                   表头介绍
##############################################
# 我们的表头的结构为
# datetime: datetime
# season: season of data in this row
# PM: PM2.5 concentration (ug/m^3)
# DEWP: Dew Point (Celsius Degree)
# TEMP: Temperature (Celsius Degree)
# HUMI: Humidity (%)
# PRES: Pressure (hPa)
# cbwd: Combined wind direction
# Iws: Cumulated wind speed (m/s)
# precipitation: hourly precipitation (mm)
# Iprec: Cumulated precipitation (mm)


##############################################
#                   数据准备
##############################################
# 合并 year, month, day, hour 列为 datetime
data_pollution['datetime'] = pd.to_datetime(data_pollution[['year', 'month', 'day', 'hour']])
data_pollution.drop(columns=['No'], inplace=True)
data_pollution.set_index('datetime', inplace=True)
# 首先 为了研究PM2.5的含量，我们注意到2013年1月1日起才开始有比较准确的测量。
# 所以我们从2013年1月1日开始进行分析
data_pollution = data_pollution.loc['2013-01-01':]

data_pollution.drop(columns=['cbwd'], inplace=True)
# 为季节创建独热向量
data_pollution = pd.get_dummies(data_pollution, columns=['season'], prefix=['season'])
data_pollution = calculate_pm25(data_pollution)  # 计算pm2.5
data_pollution = data_pollution[[col for col in data_pollution.columns if col != 'PM'] + ['PM']]

# 标签化PM值
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


data_pollution['PM_label'] = data_pollution['PM'].apply(label_pm)
#
# data_pollution.drop(columns=['PM'], inplace=True)

data_pollution.drop(columns=['hour', 'year', 'day'], inplace=True)
# 将 DataFrame 保存为 Excel 文件
data_pollution.to_excel('../data/data_pollution_prepared.xlsx', index=True)

print(data_pollution)
