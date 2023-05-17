import h5py
import pandas as pd
import re
import numpy as np

# 下面的注释为循环版本
# for i in range(0,23):
#     filename = f'timestamp_test/2023-05-13/run{i}/data.hdf5'
#     # 打开h5py文件
#     try:
#         with h5py.File(filename, 'r') as f:
#             keys = list(f.keys())
#             if keys[0] == 'SBP:FE:GMD1_Ion:Curr:AI' and keys[2] == 'mso:Area':
#                 gmd1_t = f[keys[0]][:]
#                 gmd1_timestamp_t = f[keys[1]][:]
#                 mso_area_t = f[keys[2]][:]
#                 mso_timestamp_t = f[keys[3]][:]
#             else:
#                 # 引发异常
#                 raise ValueError('keys error')
#
#         # 将四个数据集合并为一个DataFrame
#         df = pd.DataFrame({
#             'gmd1_timestamp_t': gmd1_timestamp_t,
#             'gmd1_t': gmd1_t,
#             'mso_timestamp_t': mso_timestamp_t,
#             'mso_area_t': mso_area_t,
#         })
#
#         pattern = r'timestamp_test/([0-9]+\-[0-9]+\-[0-9]+)/(run[0-9]+)/data\.hdf5'
#         match = re.search(pattern, filename)
#
#         # 将数据保存为csv文件
#         df.to_csv(match.group(1)+'-'+match.group(2)+'.csv', index=True)
#     except IOError:
#         # print(f'Error: Cannot open file "{filename}"')
#         continue
#     except ValueError:
#         continue


filename = f'timestamp_test/2023-05-13/run12/data.hdf5'
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

pattern = r'timestamp_test/([0-9]+\-[0-9]+\-[0-9]+)/(run[0-9]+)/data\.hdf5'
match = re.search(pattern, filename)

# 将数据保存为csv文件
df.to_csv(match.group(1) + '-' + match.group(2) + '.csv', index=True)
