import h5py
import numpy as np
import epics
import time
import matplotlib.pyplot as plt
import os
# 定义 EPICS PV 名称和数据存储文件名

pv_name1= "SBP:FE:GMD1_Ion:Curr:AI"
pv_name2= "mso:Area"


data_number=2000


folder= r"E:\scripts_for_commissioning\timestamp_test\\"

sub_folder=time.strftime("%Y-%m-%d")
day_folder=os.path.join(folder,sub_folder)

if not os.path.exists(day_folder):
    os.mkdir(day_folder)


run_number=0
run_folder = ''
for i in range(10000):
    run_folder=os.path.join(day_folder,"run"+str(run_number))

    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        break
    else:
        run_number+=1

filename=os.path.join(run_folder,"data.hdf5")
# 定义回调函数，保存 EPICS PV 值和时间戳到 HDF5 文件中
def data_callback1(pv_name=None, value=None, timestamp=None, **kw):
    with h5py.File(filename, 'a') as f1:
        dset = f1[pv_name]
        dset_ts = f1[pv_name + '.timestamp']

        number = len(dset)
        dset[number] = value
        dset_ts[number] = timestamp

        if number > data_number:
            pv1.auto_monitor = False
            print(pv_name + " is finished")

def data_callback2(pv_name=None, value=None, timestamp=None, **kw):
    with h5py.File(filename, 'a') as f2:
        dset = f2[pv_name]
        dset_ts = f2[pv_name + '.timestamp']

        number = len(dset)
        dset[number] = value
        dset_ts[number] = timestamp

        if number > data_number:
            pv2.auto_monitor = False
            print(pv_name + " is finished")

# 创建 HDF5 文件和数据集
with h5py.File(filename, 'w') as f:
    gmd1 = f.create_dataset(pv_name1, (data_number,), dtype=np.float64)
    gmd1_timestamp = f.create_dataset(pv_name1 + '.timestamp', (data_number,), dtype=np.float64)

    mso_area = f.create_dataset(pv_name2, (data_number,), dtype=np.float64)
    mso_timestamp = f.create_dataset(pv_name2 + '.timestamp', (data_number,), dtype=np.float64)

# 创建 EPICS PV 并注册回调函数
pv1 = epics.PV(pv_name1, auto_monitor=True, callback=data_callback1)
pv2 = epics.PV(pv_name2, auto_monitor=True, callback=data_callback2)

# 等待一段时间，让回调函数有足够的时间来收集数据
time.sleep(10)

# 读取 HDF5 文件中的数据并绘制图像
with h5py.File(filename, 'r') as f:
    gmd1_t = f[pv_name1]
    gmd1_timestamp_t = f[pv_name1 + '.timestamp']
    mso_area_t = f[pv_name2]
    mso_timestamp_t = f[pv_name2 + '.timestamp']

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0].scatter(gmd1_t[:],mso_area_t[:])
axs[0].set_xlabel(pv_name1)
axs[0].set_ylabel(pv_name2)
axs[0].set_title("correlation@0")

axs[1].scatter(np.array(gmd1_t)[:-1],np.array(mso_area_t)[1:])
axs[1].set_xlabel(pv_name1)
axs[1].set_ylabel(pv_name2)
axs[1].set_title("correlation@1")

plt.show()
