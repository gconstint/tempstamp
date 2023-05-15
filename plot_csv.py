import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# 读取CSV文件
data = pd.read_csv('delta_t2.csv', header=None, names=['filename', 'delta_t'])
pattern = r'timestamp_test/(.*?)/data\.hdf5'
# match = re.search(pattern, filename)
# 提取文件名中的run部分
x_labels = [re.search(pattern, filename).group(1) for filename in data['filename']]
print(x_labels)

x_ticks = range(len(x_labels))

# 创建图形并绘制数据
fig, ax = plt.subplots()
ax.plot(data['delta_t'])
# 将数据点绘制在折线图上
ax.scatter(x_ticks, data['delta_t'], color='red')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_xlabel('run')
ax.set_ylabel('best_delta_t')
ax.set_title('best_delta_t for each Run')

# plt.figure(figsize=(10, 6))
# plt.scatter(x_labels, data['delta_t'])
#
# plt.xticks(rotation=90)  # 旋转x轴标签
#
# plt.xlabel('filename')
# plt.ylabel('delta_t')
# plt.title('Delta_t vs. Filename')
plt.show()
