import pandas as pd

# 读取数据集
df = pd.read_csv('green_tripdata.csv')

# 将 'lpep_pickup_datetime' 列转换为日期时间格式
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])

# 设置日期为索引
df.set_index('lpep_pickup_datetime', inplace=True)

# 提取2016年12月的数据
df_month = df.loc['2016-12']

# 对数据进行10分钟重采样并统计每个时间段的记录数量
resampled_df = df_month.resample('10T').size()

# 输出路径，保存到当前文件夹
output_path = 'rides_per_10_minutes_december_2016.csv'  # 如果你想保存到当前工作目录
resampled_df.to_csv(output_path, header=['Number of Rides'])

print(f"Data saved to {output_path}")


