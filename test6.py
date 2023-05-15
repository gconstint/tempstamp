import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
以PD的数据作线性拟合，并判断gmd1和mso_area的线性关系
"""
# 读取CSV文件
df = pd.read_csv('2023-05-13-run12.csv')

# 拟合一条直线
# x = np.concatenate([df['gmd1_timestamp_t'], df['mso_timestamp_t']])
# y = np.concatenate([df['gmd1_t'], df['mso_area_t']])


x= df['mso_timestamp_t']
y = df['mso_area_t']

coefficients = np.polyfit(x, y, 1)
polyfit_fn = np.poly1d(coefficients)

x2 = df['gmd1_timestamp_t']
y2 = df['gmd1_t']

# 计算每个数据点到直线的距离
distances = np.abs(polyfit_fn(x2) - y2)

# 找到符合线性关系的数据点
threshold = 1e-7
mask = distances < threshold
linear_df = df[mask]

# 绘制符合线性关系的数据点
plt.scatter(linear_df['gmd1_timestamp_t'], linear_df['gmd1_t'], label='gmd1_t', s=1,c= 'red')
# plt.scatter(linear_df['mso_timestamp_t'], linear_df['mso_area_t'], label='mso_area_t', s=1)

# 绘制拟合的直线
plt.plot(x, polyfit_fn(x), '--', label='linear fit')

# 添加标签
plt.xlabel('timestamp')
plt.ylabel('value')
plt.title('GMD1 and MSO_AREA over time')
plt.legend()

# 显示图形
plt.show()
