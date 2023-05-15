import h5py
import numpy as np
import matplotlib.pyplot as plt

"""
test5.py是数据处理的普通版本
"""
plt.style.use("ggplot")
filename = f'timestamp_test/2023-05-13/run14/data.hdf5'

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
    # 用最佳 delta_t 进行插值
    mso_area_interp = np.interp(gmd1_timestamp_t + delta_t, mso_timestamp_t, mso_area_t)



    # # 将delta_t写入csv文件
    # with open('delta_t2.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([filename, best_delta_t])
    # 生成一个索引值列表，用于筛选数据
    index_list = np.arange(len(gmd1_t))

    # 简单的数据筛选
    threshold = 5e3
    indices = ~np.logical_and(np.abs(gmd1_t) < 1e-10, np.abs(mso_area_interp) > 2e-6)

    gmd1_tmp = gmd1_t[indices]
    gmd1_timestamp_tmp = gmd1_timestamp_t[indices]
    mso_area_tmp = mso_area_interp[indices]
    mso_timestamp_tmp = mso_timestamp_t[indices]
    print(len(gmd1_t))
    print(len(mso_area_interp))
    print(len(gmd1_timestamp_t))
    print(len(mso_timestamp_t))

    print(len(gmd1_tmp))
    print(len(mso_area_tmp))
    print(len(gmd1_timestamp_tmp))
    print(len(mso_timestamp_tmp))

    # print(gmd1_tmp==gmd1_t)

    # diffs1 = np.diff(gmd1_timestamp_tmp)
    # mean_diff1 = np.mean(diffs1)
    # mask1 = np.concatenate(([False], diffs1 <= mean_diff1))
    # gmd1_tmp = gmd1_tmp[mask1]
    # gmd1_timestamp_tmp = gmd1_timestamp_tmp[mask1]
    #
    # diffs2 = np.diff(mso_timestamp_tmp)
    # mean_diff2 = np.mean(diffs2)
    # mask2 = np.concatenate(([False], diffs2 <= mean_diff2))
    # mso_area_tmp = mso_area_tmp[mask2]
    # mso_timestamp_tmp = mso_timestamp_tmp[mask2]

    # 绘制图像
    fig = plt.figure(figsize=(6, 4))

    # ax = fig.add_subplot(2, 3, 1)
    # ax.scatter(gmd1_t, mso_area_interp, s=5, alpha=0.5)
    # ax.set_xlabel(keys[0])
    # ax.set_ylabel(keys[2])
    # ax.set_title('After data alignment,Scatter Plot of {} vs {}'.format(keys[0], keys[2]))
    # ax.legend()
    #
    # ax1 = fig.add_subplot(2, 3, 2)
    # ax1.scatter(gmd1_tmp, mso_area_tmp, s=5, alpha=0.5)
    # ax1.set_xlabel(keys[0])
    # ax1.set_ylabel(keys[2])
    # ax1.set_title('After data cleaning,Scatter Plot of {} vs {}'.format(keys[0], keys[2]))
    # ax1.legend()
    #
    # ax2 = fig.add_subplot(2, 3, 3)
    # ax2.plot(np.diff(gmd1_timestamp_t), label=keys[1])
    # ax2.plot(np.diff(mso_timestamp_t), label=keys[3])
    # ax2.set_xlabel("pulse number")
    # ax2.set_ylabel("interval")
    # ax2.set_title("interval of timestamp")
    # ax2.legend()
    #
    # ax3 = fig.add_subplot(2, 3, 4)
    # ax3.plot(np.diff(gmd1_timestamp_tmp), label=keys[1])
    # ax3.plot(np.diff(mso_timestamp_tmp), label=keys[3])
    # ax3.set_xlabel("pulse number")
    # ax3.set_ylabel("interval")
    # ax3.set_title("interval of timestamp")
    # ax3.legend()
    #
    # ax4 = fig.add_subplot(2, 3, 5)
    # ax4.plot(mso_timestamp_tmp - gmd1_timestamp_tmp, label="timestamp difference")
    # ax4.set_xlabel("pulse number")
    # ax4.set_ylabel("difference")
    # ax4.set_title("difference between\n %s \n and %s" % (keys[0], keys[2]))
    # ax4.legend()
    #
    p1 = np.polyfit(np.arange(len(gmd1_timestamp_tmp)), gmd1_timestamp_tmp, 1)
    p2 = np.polyfit(np.arange(len(mso_timestamp_tmp)), mso_timestamp_tmp, 1)
    linear_part1 = np.polyval(p1, np.arange(len(gmd1_timestamp_tmp)))
    nonlinear_part1 = gmd1_timestamp_tmp - linear_part1
    linear_part2 = np.polyval(p2, np.arange(len(mso_timestamp_tmp)))
    nonlinear_part2 = mso_timestamp_tmp - linear_part2
    print(len(nonlinear_part1))
    print(len(nonlinear_part2))
    ax6 = fig.add_subplot()
    ax6.plot(linear_part1, 'r', label='gmd1')
    ax6.plot(linear_part2, 'b', label='mso_area')
    # ax6.plot(nonlinear_part1, 'r', label='gmd1')
    # ax6.plot(nonlinear_part2, 'b', label='mso_area')
    # # ax6.plot(mso_timestamp_tmp - np.arange(np.shape(mso_timestamp_tmp)[0]) * p2[0] + p2[1], label=keys[3])
    # ax6.set_xlabel("pulse number")
    # ax6.set_ylabel("nonlinear")
    # ax6.set_title("nonlinear of timestamp")
    # ax6.legend()




    # ax5=  fig.add_subplot()
    # ax5.plot(gmd1_timestamp_tmp,gmd1_tmp*1000,label=keys[0])
    # ax5.plot(mso_timestamp_tmp,mso_area_tmp,label=keys[2])
    # ax5.legend()


    plt.show()