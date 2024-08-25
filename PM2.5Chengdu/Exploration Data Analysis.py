import pandas as pd
import matplotlib.pyplot as plt

path = "../data/data_pollution_prepared.xlsx"
data_pollution = pd.read_excel(path,index_col='datetime')
print(data_pollution)


import matplotlib.pyplot as plt

# 获取数据的列名，并去掉 `season_1` 到 `season_4` 和 `month`
columns_to_plot = [col for col in data_pollution.columns if not (col.startswith('season_') or col == 'month')]

# 获取新的列数
num_columns = len(columns_to_plot)

# 对每一列画图
plt.figure(figsize=(8, 3 * num_columns))  # 创建一个新的图形，根据列的数量动态调整图形大小
for i, group in enumerate(columns_to_plot, 1):  # 遍历需要绘制的列
    ax = plt.subplot(num_columns, 1, i)  # 创建子图，行数为列的数量，列数为1，当前子图的索引为i
    plt.plot(data_pollution.index, data_pollution[group])  # 使用日期作为 x 轴
    plt.title(group, y=0.5, loc='right')  # 设置子图标题
    plt.ylabel(group)  # 设置 y 轴标签

    # 如果不是最后一个子图，则隐藏x轴标签
    if i != num_columns:
        ax.set_xticklabels([])

plt.xlabel('Date')  # 只在最后一个子图显示 x 轴标签
plt.tight_layout()  # 调整布局以避免重叠
plt.show()  # 显示图形



