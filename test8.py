from typing import List

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
test8.py是test5.py数据处理的改进型
从test8开始程序得以改进
"""
plt.style.use("ggplot")

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



    # 寻找最佳 delta_t
    max_delta_t = 0.3  # 最大时间差 0.3
    best_delta_t = 0
    max_corr_coef = 0
    for delta_t in np.arange(-max_delta_t, max_delta_t + 0.001, 0.001):
        mso_area_interp = np.interp(gmd1_timestamp_t + delta_t, mso_timestamp_t, mso_area_t)

        corr_coef = np.corrcoef(gmd1_t, mso_area_interp)[0, 1]
        if corr_coef > max_corr_coef:
            max_corr_coef = corr_coef
            best_delta_t = delta_t
    print('best delta_t:', best_delta_t)
    print('best corr_coef:', max_corr_coef)

    # 找到了最佳的时间校准值
    gmd1_timestamp_t = gmd1_timestamp_t + best_delta_t

    # 将四个数据集合并为一个DataFrame
    df = pd.DataFrame({
        'gmd1_timestamp_t': gmd1_timestamp_t,
        'gmd1_t': gmd1_t,
        'mso_timestamp_t': mso_timestamp_t,
        'mso_area_t': mso_area_t,
    })

    # 新建一个空的DataFrame
    aligned_df = pd.DataFrame(columns=['gmd1_timestamp_t', 'gmd1_t', 'mso_timestamp_t', 'mso_area_t'])

    # 记录前一个索引的row_shift
    row_shift_pre = 1
    delete_index = []
    max_index = max(df.index)
    # 根据近似相等关系重新建立索引
    for index, gmd1_time in enumerate(df['gmd1_timestamp_t']):
        nearest_index = np.argmin(np.abs(df['mso_timestamp_t'] - gmd1_time))
        row_shift = nearest_index - index


        if row_shift == 1:
            if row_shift_pre != row_shift:
                if index - 1 > 0:
                    delete_index.append(min((index - 1),max_index))
                    print(delete_index)
            # 将当前索引的gmd1_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
            # 将当前索引的gmd1_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
            # 将下一个索引的mso_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index + 1, 'mso_timestamp_t']
            # 将下一个索引的mso_area_t值保存到aligned_df
            aligned_df.loc[index, 'mso_area_t'] = df.loc[index + 1, 'mso_area_t']
            row_shift_pre = row_shift
        elif row_shift == 2:
            if row_shift_pre != row_shift:
                if index - 1 > 0:
                    delete_index.append(min((index - 1),max_index))
                    delete_index.append(min(index,max_index))
                    print(delete_index)
            # 将当前索引的gmd1_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
            # 将当前索引的gmd1_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
            # 将下2个索引的mso_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index + 2, 'mso_timestamp_t']
            # 将下2个索引的mso_area_t值保存到aligned_df
            aligned_df.loc[index, 'mso_area_t'] = df.loc[index + 2, 'mso_area_t']

            row_shift_pre = row_shift
        elif row_shift == 3:
            if row_shift_pre != row_shift:
                if index - 1 > 0:
                    delete_index.append(min((index - 1),max_index))
                    delete_index.append(min(index, max_index))
                    delete_index.append(min((index + 1),max_index))
                    print(delete_index)
            # 将当前索引的gmd1_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
            # 将当前索引的gmd1_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
            # 将下3个索引的mso_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index + 3, 'mso_timestamp_t']
            # 将下3个索引的mso_area_t值保存到aligned_df
            aligned_df.loc[index, 'mso_area_t'] = df.loc[index + 3, 'mso_area_t']

            row_shift_pre = row_shift

        # 对于row_shift=4的情况
        elif row_shift == 4:
            if row_shift_pre!=row_shift:
                if index-1>0:
                    delete_index.append(min((index - 1),max_index))
                    delete_index.append(min(index, max_index))
                    delete_index.append(min((index + 1),max_index))
                    delete_index.append(min((index +2 ),max_index))
                    print(delete_index)

            # 将当前索引的gmd1_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
            # 将当前索引的gmd1_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
            # 将下4个索引的mso_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index + 4, 'mso_timestamp_t']
            # 将下4个索引的mso_area_t值保存到aligned_df
            aligned_df.loc[index, 'mso_area_t'] = df.loc[index + 4, 'mso_area_t']

            row_shift_pre = row_shift


        elif row_shift == 0:

            if row_shift_pre != row_shift:
                if index - 1 > 0:
                    delete_index.append(min((index - 1),max_index))
                    print(delete_index)

            # 将当前索引的gmd1_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
            # 将当前索引的gmd1_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
            # 将当前索引的mso_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index, 'mso_timestamp_t']
            # 将当前索引的mso_area_t值保存到aligned_df
            aligned_df.loc[index, 'mso_area_t'] = df.loc[index, 'mso_area_t']

            row_shift_pre = row_shift
        elif row_shift == -1:
            if row_shift_pre != row_shift:
                if index - 1>0:
                    delete_index.append(min((index - 1),max_index))
                    print(delete_index)


            # 将当前索引的gmd1_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
            # 将当前索引的gmd1_t值保存到aligned_df
            aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
            # 将上一个索引的mso_timestamp_t值保存到aligned_df
            aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index , 'mso_timestamp_t']
            # 将上一个索引的mso_area_t值保存到aligned_df
            aligned_df.loc[index, 'mso_area_t'] = df.loc[index , 'mso_area_t']

            row_shift_pre = row_shift

        print(f'Row shift for gmd1_timestamp_t at index {index}: {row_shift}')

    # 删除第一行
    # df.dropna(inplace=True)
    # 重设索引
    # df.reset_index(drop=True, inplace=True)
    # 保存为csv文件,显示索引
    # df.to_csv('data2.csv', index=True)

    # # 重设索引
    # # aligned_df.reset_index(drop=True, inplace=True)

    aligned_df.drop(index= delete_index,inplace=True)
    print("delete index:",delete_index)
    # idx = np.where(abs(aligned_df['gmd1_timestamp_t'] - aligned_df['mso_timestamp_t']) > 0.15)
    # print(idx)

    # 找到相邻的mso_timestamp_t差值大于0.3的索引号
    idx = np.where(abs(np.diff(aligned_df['mso_timestamp_t'])) > 0.4)[0]+np.array(1)
    print(idx)
    # 打印结果
    print(aligned_df.iloc[idx])

    aligned_df.to_csv(f'aligned_data{number}.csv', index=True)


    # 绘制符合要求的gmd1_t数据点
    fig = plt.figure(figsize=(6, 4))
    fig.suptitle(filename)
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
    ax4.plot(np.diff(aligned_df['mso_timestamp_t']), label= "pd.timestamp")
    ax4.set_xlabel("pulse number")
    ax4.set_ylabel("interval")
    ax4.set_title("interval of timestamp")
    ax4.legend()

    ax5 = fig.add_subplot(1, 4, 3)
    ax5.plot( aligned_df['mso_timestamp_t']-aligned_df['gmd1_timestamp_t'] , label="timestamp difference")
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
    ax6.plot( nonlinear_part1,label=  "gmd.timestamp")
    ax6.plot(nonlinear_part2, label= "pd.timestamp")

    ax6.set_xlabel("pulse number")
    ax6.set_ylabel("nonlinear")
    ax6.set_title("nonlinear of timestamp")
    ax6.legend()

    plt.show()
    # 保存图像
    # plt.savefig(f'aligned_data{number}.png')
