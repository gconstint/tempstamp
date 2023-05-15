import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

pv_name1="SBP:FE:GMD1_Ion:Curr:AI"
pv_name2="mso:Area"
filename = 'timestamp_test/2023-05-13/run22/data.hdf5'

# 读取 HDF5 文件中的数据
with h5py.File(filename, 'r') as f:
    gmd1_t = f[pv_name1]
    gmd1_timestamp_t = f[pv_name1 + '.timestamp']
    mso_area_t = f[pv_name2]
    mso_timestamp_t = f[pv_name2 + '.timestamp']

    # 寻找最佳 delta_t
    max_delta_t = 0.3  # 最大时间差
    best_delta_t = 0
    max_corr_coef = 0
    for delta_t in np.arange(-max_delta_t, max_delta_t + 0.001, 0.001):
        mso_area_interp = np.interp(gmd1_timestamp_t + delta_t, mso_timestamp_t, mso_area_t)
        corr_coef = np.corrcoef(gmd1_t, mso_area_interp)[0, 1]
        if corr_coef > max_corr_coef:
            max_corr_coef = corr_coef
            best_delta_t = delta_t

    print('best delta_t:', best_delta_t)

    # 用最佳 delta_t 进行插值
    mso_area_interp = np.interp(gmd1_timestamp_t + delta_t, mso_timestamp_t, mso_area_t)


    threshold = 5e3

    indices = np.abs(mso_area_interp /gmd1_t) > threshold

    gmd1_tmp = gmd1_t[indices]
    gmd1_timestamp_tmp = gmd1_timestamp_t[indices]

    mso_area_tmp = mso_area_interp[indices]
    mso_timestamp_tmp = mso_timestamp_t[indices]


    # 绘制散点图
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 3, 1)
    # ax.scatter(gmd1_t_shifted_filtered,mso_area_t_filtered, s=5, alpha=0.5)
    ax.scatter(gmd1_t, mso_area_interp, s=5, alpha=0.5)
    ax.set_xlabel(pv_name1)
    ax.set_ylabel(pv_name2)
    ax.set_title('Scatter Plot of {} vs {}'.format(pv_name1, pv_name2))

    ax2=fig.add_subplot(1,3,2)
    # 作pv1的时间和值的散点图
    ax2.scatter(gmd1_timestamp_tmp, gmd1_tmp, s=5, alpha=0.5,label='gmd1')
    ax2.scatter(mso_timestamp_tmp, mso_area_tmp, s=5, alpha=0.5,label='mso_area')

    ax2.set_xlabel('time')
    ax2.set_ylabel(pv_name1)
    ax2.set_title('Scatter Plot of {} vs time'.format(pv_name1))
    ax2.legend()
    ax3=fig.add_subplot(1,3,3)
    # 作pv2的时间和值的散点图
    ax3.scatter(mso_timestamp_tmp, mso_area_tmp, s=5, alpha=0.5)
    ax3.set_xlabel('time')
    ax3.set_ylabel(pv_name2)
    ax3.set_title('Scatter Plot of {} vs time'.format(pv_name2))

    plt.show()
