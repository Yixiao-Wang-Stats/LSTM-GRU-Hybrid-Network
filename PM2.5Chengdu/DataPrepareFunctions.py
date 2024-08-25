import pandas as pd


def calculate_pm25(data_pollution):
    # 具体的填补方式是存在一个以上则求平均值，否则用线性差值解决
    data_pollution['PM'] = None

    for index, row in data_pollution.iterrows():
        PM_values = [row['PM_Caotangsi'], row['PM_Shahepu'], row['PM_US Post']]
        PM_exist = [not pd.isnull(pm) for pm in PM_values]  # 检查每个 PM 值是否存在

        if any(PM_exist):
            data_pollution.at[index, 'PM'] = sum(pm for pm, exists in zip(PM_values, PM_exist) if exists) / sum(
                PM_exist)
    data_pollution.drop(columns=['PM_Caotangsi', 'PM_Shahepu', 'PM_US Post'], inplace=True)
    # 对所有项线性插值
    for name in data_pollution.columns:
        data_pollution[name] = pd.to_numeric(data_pollution[name], errors='coerce')
        data_pollution[name] = data_pollution[name].interpolate(method='linear')

    return data_pollution