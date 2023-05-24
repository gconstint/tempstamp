import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

"""
test14是test14的改进型，进一步优化了速度问题，以及代码规范
"""
number = 12

filename = f'timestamp_test/2023-05-13/run{number}/data.hdf5'

with h5py.File(filename, 'r') as f:
    keys = list(f.keys())
    if keys[0] == 'SBP:FE:GMD1_Ion:Curr:AI' and keys[2] == 'mso:Area':
        gmd1_t = f[keys[0]][:]
        gmd1_timestamp_t = f[keys[1]][:]
        mso_area_t = f[keys[2]][:]
        mso_timestamp_t = f[keys[3]][:]

    else:
        # 引发异常
        raise ValueError('keys error')

# 将四个数据集合并为一个DataFrame
df = pd.DataFrame({
    'gmd1_timestamp_t': gmd1_timestamp_t,
    'gmd1_t': gmd1_t,
    'mso_timestamp_t': mso_timestamp_t,
    'mso_area_t': mso_area_t,
})

# 寻找最佳 delta_t
max_delta_t = 0.3  # 最大时间差 0.3
best_delta_t = 0
max_corr_coef = 0
aligned_df = pd.DataFrame()

mso_timestamp_t_copy = df['mso_timestamp_t'].copy()  # 创建 mso_timestamp_t 的副本
mso_timestamp_t_values = df['mso_timestamp_t'].values
gmd1_timestamp_t_values = df['gmd1_timestamp_t'].values

for delta_t in np.arange(-max_delta_t, max_delta_t + 0.001, 0.001):
    aligned_df['gmd1_timestamp_t'] = gmd1_timestamp_t_values

    differences = np.abs(mso_timestamp_t_values[:, np.newaxis] - aligned_df['gmd1_timestamp_t'].values - delta_t)
    nearest_indices = np.argmin(differences, axis=0)

    aligned_df['gmd1_t'] = df['gmd1_t']
    aligned_df['mso_timestamp_t'] = df['mso_timestamp_t'].values[nearest_indices]
    aligned_df['mso_area_t'] = df['mso_area_t'].values[nearest_indices]

    aligned_df.dropna(subset=['gmd1_t', 'mso_area_t'], inplace=True)

    corr_coef = pearsonr(aligned_df['gmd1_t'], aligned_df['mso_area_t'])[0]

    if corr_coef > max_corr_coef:
        max_corr_coef = corr_coef
        best_delta_t = delta_t

    print('delta_t:', delta_t, 'corr_coef:', corr_coef)

print('best delta_t:', best_delta_t)
print('best corr_coef:', max_corr_coef)


row_shift_pre = 1
delete_index = []
max_index = max(df.index)

for index, gmd1_time in enumerate(gmd1_timestamp_t_values):
    gmd1_time = gmd1_time + best_delta_t
    nearest_index = np.argmin(np.abs(mso_timestamp_t_values - gmd1_time))
    row_shift = nearest_index - index
    if row_shift_pre != row_shift and index - 1 > 0:
        delete_index.extend([index - 1 + i for i in range(np.abs(row_shift_pre - row_shift) + 1) if index - 1 + i <= max_index])

    if np.min(np.abs(mso_timestamp_t_values - gmd1_time)) < 0.05:
        aligned_df['gmd1_timestamp_t'].values[index] = gmd1_timestamp_t_values[index]
        aligned_df['gmd1_t'].values[index] = df['gmd1_t'].values[index]
        aligned_df['mso_timestamp_t'].values[index] = df['mso_timestamp_t'].values[nearest_index]
        aligned_df['mso_area_t'].values[index] = df['mso_area_t'].values[nearest_index]
        row_shift_pre = row_shift

aligned_df['gmd1_timestamp_t'] = aligned_df['gmd1_timestamp_t'] + best_delta_t
aligned_df = aligned_df[~aligned_df.index.isin(delete_index)]

# 绘制符合要求的gmd1_t数据点
fig = plt.figure(figsize=(6, 4))
fig.suptitle('test13' + filename)
ax = fig.add_subplot(1, 4, 1)
# ax.plot(aligned_df['gmd1_timestamp_t'], aligned_df['gmd1_t'], label='(new)gmd1_t', c='red')
ax.scatter(aligned_df['gmd1_t'], aligned_df['mso_area_t'], label='gmd1_t vs mso_area_t', c='red')
# 添加标签
# ax.set_xlabel('timestamp')
# ax.set_ylabel('value')
# ax.set_title('GMD1 and MSO_AREA over time')
ax.legend()

ax4 = fig.add_subplot(1, 4, 2)
ax4.plot(np.diff(aligned_df['gmd1_timestamp_t']), label="gmd.timestamp")
ax4.plot(np.diff(aligned_df['mso_timestamp_t']), label="pd.timestamp")
ax4.set_xlabel("pulse number")
ax4.set_ylabel("interval")
ax4.set_title("interval of timestamp")
ax4.legend()

ax5 = fig.add_subplot(1, 4, 3)
ax5.plot(aligned_df['mso_timestamp_t'] - aligned_df['gmd1_timestamp_t'], label="timestamp difference")
ax5.set_xlabel("pulse number")
ax5.set_ylabel("difference")
ax5.set_title("difference between %s and %s" % ("pd", "gmd"))
ax5.legend()

index1 = aligned_df['gmd1_timestamp_t'].index
index2 = aligned_df['mso_timestamp_t'].index
# print(index1)
# print(range(len(index1)))
p1 = np.polyfit(index1, aligned_df['gmd1_timestamp_t'].astype(float), 1)
p2 = np.polyfit(index2, aligned_df['mso_timestamp_t'].astype(float), 1)
linear_part1 = np.polyval(p1, index1)
nonlinear_part1 = aligned_df['gmd1_timestamp_t'] - linear_part1
linear_part2 = np.polyval(p2, index2)
nonlinear_part2 = aligned_df['mso_timestamp_t'] - linear_part2

ax6 = fig.add_subplot(1, 4, 4)
ax6.plot(nonlinear_part1, label="gmd.timestamp")
ax6.plot(nonlinear_part2, label="pd.timestamp")

ax6.set_xlabel("pulse number")
ax6.set_ylabel("nonlinear")
ax6.set_title("nonlinear of timestamp")
ax6.legend()

plt.show()
# 保存图像
# plt.savefig(f'aligned_data{number}.png')