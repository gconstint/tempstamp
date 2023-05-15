import h5py
import pylab as plt
import numpy as np
from scipy.interpolate import interp1d
# from scipy import interpolate
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

plt.style.use("ggplot")

pv_name1="SBP:FE:GMD1_Ion:Curr:AI"
pv_name2="mso:Area"
filename = 'data.hdf5'

# # 读取 HDF5 文件中的数据并绘制图像
# with h5py.File(filename, 'r') as f:
#     gmd1_t = f[pv_name1][:]
#     gmd1_timestamp_t = f[pv_name1 + '.timestamp'][:]
#     mso_area_t = f[pv_name2][:]
#     mso_timestamp_t = f[pv_name2 + '.timestamp'][:]
#
#     # # 用插值法将两个 PV 的时间戳对齐
#     # gmd1_timestamp_t = gmd1_timestamp_t - gmd1_timestamp_t[0]
#     # mso_timestamp_t = mso_timestamp_t - mso_timestamp_t[0]
#     # f1 = interpolate.interp1d(gmd1_timestamp_t, gmd1_t, kind='linear', fill_value='extrapolate')
#     # gmd1_t = f1(mso_timestamp_t)
#
#
#     # # 用插值法将两个 PV 的时间戳对准到0.2s以内
#     # gmd1_timestamp_t = gmd1_timestamp_t - gmd1_timestamp_t[0]
#     # mso_timestamp_t = mso_timestamp_t - mso_timestamp_t[0]
#     # gmd1_t_interp = []
#     # gmd1_timestamp_t_interp = []
#     # mso_area_t_interp = []
#     # mso_timestamp_t_interp = []
#     # for i in range(len(mso_timestamp_t)):
#     #         if abs(mso_timestamp_t[i] - gmd1_timestamp_t[i]) < 0.2:
#     #             gmd1_t_interp.append(gmd1_t[i])
#     #             gmd1_timestamp_t_interp.append(gmd1_timestamp_t[i])
#     #             mso_area_t_interp.append(mso_area_t[i])
#     #             mso_timestamp_t_interp.append(mso_timestamp_t[i])
#     #
#     # gmd1_t_interp = np.array(gmd1_t_interp)
#     # gmd1_timestamp_t_interp = np.array(gmd1_timestamp_t_interp)
#     # mso_area_t_interp = np.array(mso_area_t_interp)
#     # mso_timestamp_t_interp = np.array(mso_timestamp_t_interp)
#
#     # 用插值法将两个 PV 的时间戳对齐
#     gmd1_timestamp_t = gmd1_timestamp_t - gmd1_timestamp_t[0]
#     mso_timestamp_t = mso_timestamp_t - mso_timestamp_t[0]
#     f1 = interpolate.interp1d(gmd1_timestamp_t, gmd1_t, kind='linear', fill_value='extrapolate')
#     gmd1_t_interp = f1(mso_timestamp_t)
#     gmd1_timestamp_t_interp = mso_timestamp_t
#     mso_area_t_interp = mso_area_t
#     mso_timestamp_t_interp = mso_timestamp_t
#
#
#
#     fig = plt.figure(figsize=(15, 8), dpi=150)
#
#     ax1 = fig.add_subplot(2, 3, 1)
#     ax1.scatter(gmd1_t_interp,mso_area_t_interp)
#     ax1.set_title(pv_name1)
#
#
#     ax2 = fig.add_subplot(2, 3, 2)
#     ax2.scatter(gmd1_t,mso_area_t)
#     ax2.set_title(pv_name2)
#
#     # ax4 = fig.add_subplot(2, 3, 3)
#     # ax4.plot(np.diff(gmd1_timestamp_t_interp), label=pv_name1 + ".timestamp")
#     # ax4.plot(np.diff(mso_timestamp_t_interp), label=pv_name2 + ".timestamp")
#     # ax4.set_xlabel("pulse number")
#     # ax4.set_ylabel("interval")
#     # ax4.set_title("interval of timestamp")
#     # ax4.legend()
#
#     ax5 = fig.add_subplot(2, 3, 3)
#     ax5.plot( gmd1_timestamp_t_interp- mso_timestamp_t_interp, label="timestamp difference")
#     ax5.set_xlabel("pulse number")
#     ax5.set_ylabel("difference")
#     ax5.set_title("difference between\n %s \n and %s" % (pv_name2, pv_name1))
#     ax5.legend()
#     plt.tight_layout()
#
#     plt.show()
#     fig.savefig(filename+".png",dpi=600)
# 读取 HDF5 文件中的数据
# 读取 HDF5 文件中的数据并绘制图像
with h5py.File(filename, 'r') as f:
    gmd1_t = f[pv_name1]
    gmd1_timestamp_t = f[pv_name1 + '.timestamp']
    mso_area_t = f[pv_name2]
    mso_timestamp_t = f[pv_name2 + '.timestamp']

    # # 对时间戳进行插值，使得两个 PV 的时间戳相同
    # gmd1_interp = interp1d(gmd1_timestamp_t, gmd1_t, kind='linear', bounds_error=False)
    # mso_area_interp = interp1d(mso_timestamp_t, mso_area_t, kind='linear', bounds_error=False)
    # gmd1_t_aligned = gmd1_interp(mso_timestamp_t)
    # mso_area_t_aligned = mso_area_interp(gmd1_timestamp_t)

    # 对时间戳进行插值，使得两个 PV 的时间戳相同
    t_common = np.arange(max(gmd1_timestamp_t[0], mso_timestamp_t[0]), min(gmd1_timestamp_t[-1], mso_timestamp_t[-1]), 0.2)

    gmd1_interp = np.interp(t_common, gmd1_timestamp_t, gmd1_t)
    mso_area_interp = np.interp(t_common, mso_timestamp_t, mso_area_t)
    mso_timestamp_interp = np.interp(t_common, mso_timestamp_t, mso_timestamp_t)

    # 绘制散点图
    fig = plt.figure(figsize=(15, 8), dpi=150)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(gmd1_t_aligned, mso_area_t_aligned)
    ax1.set_title(pv_name1 + ' vs ' + pv_name2)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(gmd1_t,mso_area_t)
    ax2.set_title(pv_name2)

    ax4 = fig.add_subplot(2, 3, 3)
    ax4.plot(np.diff(gmd1_timestamp_t), label=pv_name1 + ".timestamp")
    ax4.plot(np.diff(mso_timestamp_t), label=pv_name2 + ".timestamp")
    ax4.set_xlabel("pulse number")
    ax4.set_ylabel("interval")
    ax4.set_title("interval of timestamp")
    ax4.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(filename+".png",dpi=600)