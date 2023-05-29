#-*- coding: utf-8 -*-
import re
import time

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pydm import Display
import sys,os
from os import path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class WorkerThread(QThread):
    progress_update = pyqtSignal(int)  # 用于通知主线程进度更新的信号
    finished = pyqtSignal()
    def __init__(self,plot,textbox_delta_t,textbox_corr_coef,textbox_fitting,filename,linear_flag,max_delta_t,step):
        super(WorkerThread, self).__init__()
        self.is_running = True
        self.plot = plot
        self.textbox1 = textbox_delta_t
        self.textbox2 = textbox_corr_coef
        self.textbox3 = textbox_fitting
        self.filename = filename
        self.linear_flag = linear_flag
        self.max_delta_t = max_delta_t
        self.step = step


    def run(self):
        filename = self.filename
        linear_flag = self.linear_flag
        try:
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
        except ValueError:
            pass
        else:
            # 将四个数据集合并为一个DataFrame
            df = pd.DataFrame({
                'gmd1_timestamp_t': gmd1_timestamp_t,
                'gmd1_t': gmd1_t,
                'mso_timestamp_t': mso_timestamp_t,
                'mso_area_t': mso_area_t,
            })

            # 寻找最佳 delta_t
            # max_delta_t = float(self.lineEdit_3.text()) * float(self.lineEdit_4.text())  # 最大时间差 0.3
            # step = float(self.lineEdit_9.text())  # 步长 0.001
            max_delta_t = self.max_delta_t
            step = self.step

            best_delta_t = 0
            max_corr_coef = 0
            corr_coef = 0
            # aligned_df = pd.DataFrame()
            aligned_df = pd.DataFrame(columns=['gmd1_timestamp_t', 'gmd1_t', 'mso_timestamp_t', 'mso_area_t'])
            mso_timestamp_t_copy = df['mso_timestamp_t'].copy()  # 创建 mso_timestamp_t 的副本
            mso_timestamp_t_values = mso_timestamp_t_copy.values
            gmd1_timestamp_t_values = df['gmd1_timestamp_t'].values

            for delta_t in np.arange(-max_delta_t, max_delta_t + step, step):
                aligned_df['gmd1_timestamp_t'] = df['gmd1_timestamp_t']

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
                # 保留一位小数
                self.progress_update.emit(int((delta_t + max_delta_t) / (2 * max_delta_t) * 100))
                # print('delta_t:', delta_t, 'corr_coef:', corr_coef)

            # print('best delta_t:', best_delta_t)
            # print('best corr_coef:', max_corr_coef)
            # 保留小数点后10位
            best_delta_t = round(best_delta_t, 10)
            max_corr_coef = round(max_corr_coef, 10)

            self.textbox1.setText(str(best_delta_t))
            self.textbox2.setText(str(max_corr_coef))
            row_shift_pre = 1
            delete_index = []
            max_index = max(df.index)

            for index, gmd1_time in enumerate(gmd1_timestamp_t_values):
                gmd1_time = gmd1_time + best_delta_t
                nearest_index = np.argmin(np.abs(mso_timestamp_t_values - gmd1_time))
                row_shift = nearest_index - index
                if row_shift_pre != row_shift and index - 1 > 0:
                    delete_index.extend([index - 1 + i for i in range(np.abs(row_shift_pre - row_shift) + 1) if index - 1 + i <= max_index])

                aligned_df['gmd1_timestamp_t'].values[index] = gmd1_timestamp_t_values[index]
                aligned_df['gmd1_t'].values[index] = df['gmd1_t'].values[index]
                aligned_df['mso_timestamp_t'].values[index] = df['mso_timestamp_t'].values[nearest_index]
                aligned_df['mso_area_t'].values[index] = df['mso_area_t'].values[nearest_index]
                row_shift_pre = row_shift

            aligned_df['gmd1_timestamp_t'] = aligned_df['gmd1_timestamp_t'] + best_delta_t

            if linear_flag:
                # 计算线性回归模型
                slope, intercept, _, _, _ = linregress(aligned_df['gmd1_t'].tolist(), aligned_df['mso_area_t'].tolist())

                # 计算残差
                residuals = aligned_df['mso_area_t'] - (slope * aligned_df['gmd1_t'] + intercept)

                # 设置拟合残差阈值，超过阈值的行将被删除
                threshold = 1 * np.std(residuals)  # 可根据需要调整阈值

                # 删除拟合残差超过阈值的行
                aligned_df = aligned_df[np.abs(residuals) < threshold]

                # 重新计算相关系数
                corr_coef = pearsonr(aligned_df['gmd1_t'].tolist(), aligned_df['mso_area_t'].tolist())[0]

                # 打印相关系数结果
                # print('Original correlation coefficient:', corr_coef)
                # print(len(aligned_df))
                # 保留小数点后10位
                corr_coef = round(corr_coef, 10)
                self.textbox3.setText(str(corr_coef))
            else:
                aligned_df = aligned_df[~aligned_df.index.isin(delete_index)]

            try:
                # 读取CSV文件为DataFrame
                df_csv = pd.read_csv('data_traverse.csv')
            except FileNotFoundError:
                # 如果不存在csv文件，则创建一个df
                df_csv = pd.DataFrame(columns=['filename', 'delta_t', 'corr_coef'])
                df_csv.to_csv('data_traverse.csv', index=False)
                # 找到与filename匹配的行，并进行删除操作
            df_csv = df_csv[~(df_csv['filename'] == filename)].reset_index(drop=True)
            # 在df_csv中添加一行
            df_csv.loc[len(df_csv)] = [filename, best_delta_t, corr_coef]
            # 将更新后的DataFrame写回文件
            df_csv.to_csv('data_traverse.csv', index=False)

            self.plot(aligned_df)

            # 任务完成后发送信号
            self.finished.emit()
class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super(MyDisplay, self).__init__(parent=parent, args=args, macros=macros)

        self.fig1 = Figure(facecolor=(1, 1, 1, 1))
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor((255 / 255, 255 / 255, 255 / 255, 255 / 255))
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)

        self.ax1.set_title('Fig1')
        self.fig1.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.9, hspace=0.1, wspace=0.1)
        self.canvas1 = FigureCanvas(self.fig1)
        # self.toolbar1 = NavigationToolbar(self.canvas1, self)
        # # add Canvas to Layout
        self.ui.PyDMEmbeddedDisplay_1.layout.addWidget(self.canvas1)
        # self.ui.PyDMEmbeddedDisplay_1.layout.addWidget(self.toolbar1)

        self.fig2 = Figure(facecolor=(1, 1, 1, 1))
        self.ax2 = self.fig2.add_subplot(3,1,1)
        self.ax2.set_facecolor((255 / 255, 255 / 255, 255 / 255, 255 / 255))
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.set_title('Fig2')

        self.ax3 = self.fig2.add_subplot(3, 1, 2)
        self.ax3.set_facecolor((255 / 255, 255 / 255, 255 / 255, 255 / 255))
        self.ax3.spines['top'].set_visible(False)
        self.ax3.spines['right'].set_visible(False)
        self.ax3.set_title('Fig3')

        self.ax4 = self.fig2.add_subplot(3, 1, 3)
        self.ax4.set_facecolor((255 / 255, 255 / 255, 255 / 255, 255 / 255))
        self.ax4.spines['top'].set_visible(False)
        self.ax4.spines['right'].set_visible(False)
        self.ax4.set_title('Fig4')

        self.fig2.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.9, hspace=0.7, wspace=0.1)
        self.canvas2 = FigureCanvas(self.fig2)
        # self.toolbar2 = NavigationToolbar(self.canvas2, self)
        # add Canvas to Layout
        self.ui.PyDMEmbeddedDisplay_2.layout.addWidget(self.canvas2)
        # self.ui.PyDMEmbeddedDisplay_2.layout.addWidget(self.toolbar2)

        self.worker_thread =None
        self.pushButton_4.clicked.connect(self.start1)
        self.pushButton_3.clicked.connect(self.start_thread)

        self.pushButton_5.clicked.connect(self.plot_csv)

        self.textbox_1 = self.lineEdit_2
        self.pushButton_2.clicked.connect(self.on_button_clicked_1)
        self.filename = None

    def ui_filename(self):
        return 'gmd_mso.ui'

    def ui_filepath(self):
        return path.join(path.dirname(path.realpath(__file__)), self.ui_filename())

    def on_button_clicked_1(self):
        # show folder selection dialog box
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Select file", "/Users/guanzhh/PycharmProjects/tempstamp/timestamp_test/2023-05-13/", options=options)

        # If the user selects a file, set its path to the text of the text box
        if file_path:
            self.filename = file_path
            # file_name = os.path.basename(file_path)
            # self.textbox_1.setText(file_name)

            # file_name = os.path.basename(file_path)
            parent_dir, last_dir = os.path.split(os.path.dirname(file_path))
            _, second_last_dir = os.path.split(parent_dir)
            final_dirs = os.path.join(second_last_dir, last_dir)
            self.textbox_1.setText(final_dirs)
    def get_folder_path_1(self):
        # returns the text of the text box
        return self.textbox_1.text()
    def plot(self,aligned_df):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()

        self.ax1.scatter(aligned_df['gmd1_t'], aligned_df['mso_area_t'], label='gmd1 vs mso_area', c='red')
        # 添加标签
        self.ax1.set_xlabel('timestamp')
        self.ax1.set_ylabel('value')
        self.ax1.set_title('GMD1 and MSO_AREA over timestamps')
        self.ax1.legend(loc='upper right')

        self.ax2.plot(np.diff(aligned_df['gmd1_timestamp_t']), label="gmd.timestamp")
        self.ax2.plot(np.diff(aligned_df['mso_timestamp_t']), label="pd.timestamp")
        self.ax2.set_xlabel("pulse number")
        self.ax2.set_ylabel("interval")
        self.ax2.set_title("interval of timestamp")
        self.ax2.legend(loc='upper right')

        self.ax3.plot(aligned_df['mso_timestamp_t'] - aligned_df['gmd1_timestamp_t'], label="timestamp difference")
        self.ax3.set_xlabel("pulse number")
        self.ax3.set_ylabel("difference")
        self.ax3.set_title("difference between %s and %s" % ("pd", "gmd"))
        self.ax3.legend(loc='upper right')

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

        self.ax4.plot(nonlinear_part1, label="gmd.timestamp")
        self.ax4.plot(nonlinear_part2, label="pd.timestamp")

        self.ax4.set_xlabel("pulse number")
        self.ax4.set_ylabel("nonlinear")
        self.ax4.set_title("nonlinear of timestamp")
        # 在右上角添加图例
        self.ax4.legend(loc='upper right')

        self.canvas1.draw()
        self.canvas2.draw()
    def start1(self):
        filename = self.filename
        linear_flag = self.checkBox.isChecked()
        try:
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
        except ValueError:
            pass
        else:


            # 寻找最佳 delta_t
            max_delta_t = float(self.lineEdit_3.text()) * float(self.lineEdit_4.text())  # 最大时间差 0.3
            best_delta_t = 0
            max_corr_coef = 0
            corr_coef = 0
            step = float(self.lineEdit_9.text())  # 步长 0.001

            for delta_t in np.arange(-max_delta_t, max_delta_t + step, step):
                mso_area_interp = np.interp(gmd1_timestamp_t + delta_t, mso_timestamp_t, mso_area_t)

                corr_coef = np.corrcoef(gmd1_t, mso_area_interp)[0, 1]
                if corr_coef > max_corr_coef:
                    max_corr_coef = corr_coef
                    best_delta_t = delta_t
            # print('best delta_t:', best_delta_t)
            # print('best corr_coef:', max_corr_coef)
            # 保留小数点后10位
            best_delta_t = round(best_delta_t, 10)
            max_corr_coef = round(max_corr_coef, 10)
            self.lineEdit_8.setText(str(best_delta_t))
            self.lineEdit_7.setText(str(max_corr_coef))

            # 找到了最佳的时间校准值
            gmd1_timestamp_t = gmd1_timestamp_t + best_delta_t

            # 将四个数据集合并为一个DataFrame
            df = pd.DataFrame({
                'gmd1_timestamp_t': gmd1_timestamp_t,
                'gmd1_t': gmd1_t,
                'mso_timestamp_t': mso_timestamp_t,
                'mso_area_t': mso_area_t,
            })
            # aligned_df = pd.DataFrame()
            aligned_df = pd.DataFrame(columns=['gmd1_timestamp_t', 'gmd1_t', 'mso_timestamp_t', 'mso_area_t'])
            # 记录前一个索引的row_shift
            row_shift_pre = 1
            delete_index = []
            max_index = max(df.index)
            # 根据近似相等关系重新建立索引
            for index, gmd1_time in enumerate(df['gmd1_timestamp_t']):
                nearest_index = np.argmin(np.abs(df['mso_timestamp_t'] - gmd1_time))
                row_shift = nearest_index - index

                if row_shift_pre != row_shift and index - 1 > 0:
                    delete_index.extend([index - 1 + i for i in range(np.abs(row_shift_pre - row_shift) + 1) if index - 1 + i <= max_index])

                # 将当前索引的gmd1_timestamp_t值保存到aligned_df
                aligned_df.loc[index, 'gmd1_timestamp_t'] = df.loc[index, 'gmd1_timestamp_t']
                # 将当前索引的gmd1_t值保存到aligned_df
                aligned_df.loc[index, 'gmd1_t'] = df.loc[index, 'gmd1_t']
                # 将下row_shift个索引的mso_timestamp_t值保存到aligned_df
                aligned_df.loc[index, 'mso_timestamp_t'] = df.loc[index + row_shift, 'mso_timestamp_t']
                # 将下row_shift个索引的mso_area_t值保存到aligned_df
                aligned_df.loc[index, 'mso_area_t'] = df.loc[index + row_shift, 'mso_area_t']
                row_shift_pre = row_shift
                ### 方法1
            if linear_flag:
                # 计算线性回归模型
                slope, intercept, _, _, _ = linregress(aligned_df['gmd1_t'].tolist(), aligned_df['mso_area_t'].tolist())

                # 计算残差
                residuals = aligned_df['mso_area_t'] - (slope * aligned_df['gmd1_t'] + intercept)

                # 设置拟合残差阈值，超过阈值的行将被删除
                threshold = 1 * np.std(residuals)  # 可根据需要调整阈值

                # 删除拟合残差超过阈值的行
                aligned_df = aligned_df[np.abs(residuals) < threshold]

                # 重新计算相关系数
                corr_coef = pearsonr(aligned_df['gmd1_t'].tolist(), aligned_df['mso_area_t'].tolist())[0]

                # 打印相关系数结果
                # print('Original correlation coefficient:', corr_coef)
                # print(len(aligned_df))
                # 保留小数点后10位
                corr_coef = round(corr_coef, 10)
                self.lineEdit_11.setText(str(corr_coef))
            else:
                aligned_df = aligned_df[~aligned_df.index.isin(delete_index)]
            # 将delta_t写入csv文件
            # 覆盖重复项
            # with open('data_interp.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([filename, best_delta_t,corr_coef])

            try:
                # 读取CSV文件为DataFrame
                df_csv = pd.read_csv('data_interp.csv')
            except FileNotFoundError:
                # 如果不存在csv文件，则创建一个df
                df_csv = pd.DataFrame(columns=['filename', 'delta_t', 'corr_coef'])
                df_csv.to_csv('data_interp.csv', index=False)
            # 找到与filename匹配的行，并进行删除操作
            df_csv = df_csv[~(df_csv['filename'] == filename)].reset_index(drop=True)
            # 在df_csv中添加一行
            df_csv.loc[len(df_csv)] = [filename, best_delta_t, corr_coef]
            # 将更新后的DataFrame写回文件
            df_csv.to_csv('data_interp.csv', index=False)

            self.plot(aligned_df)
    def start_thread(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return
        # 先停止当前运行的线程（如果有）
        self.stop_thread()
        # 删除先前的线程对象
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        filename = self.filename
        linear_flag = self.checkBox.isChecked()
        max_delta_t = float(self.lineEdit_3.text()) * float(self.lineEdit_4.text())  # 最大时间差 0.3
        step = float(self.lineEdit_9.text())  # 步长 0.001

        # 创建并启动工作线程
        self.worker_thread = WorkerThread(self.plot,self.lineEdit_5,self.lineEdit_6,self.lineEdit_10,filename,linear_flag,max_delta_t,step,)
        self.worker_thread.progress_update.connect(self.update_progress)
        self.worker_thread.finished.connect(self.thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.quit)  # 线程完成后停止线程
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)  # 清理资源
        self.worker_thread.start()

    def stop_thread(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()  # 等待线程停止
    def update_progress(self, progress):
        # 更新进度信息
        # self.PyDMLabel.setText("running")
        self.progressBar.setValue(progress)

    def thread_finished(self):
        # 线程完成时的处理，例如进行清理操作
        self.progressBar.setValue(100)
        self.worker_thread.deleteLater()
        self.worker_thread = None  # 重置线程对象
        time.sleep(1)
        self.progressBar.setValue(0)
    def start2(self):
        filename = self.filename
        linear_flag = self.checkBox.isChecked()
        try:
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
        except ValueError:
            pass
        else:
            # 将四个数据集合并为一个DataFrame
            df = pd.DataFrame({
                'gmd1_timestamp_t': gmd1_timestamp_t,
                'gmd1_t': gmd1_t,
                'mso_timestamp_t': mso_timestamp_t,
                'mso_area_t': mso_area_t,
            })

            # 寻找最佳 delta_t
            max_delta_t = float(self.lineEdit_3.text()) * float(self.lineEdit_4.text()) # 最大时间差 0.3
            step = float(self.lineEdit_9.text()) # 步长 0.001
            best_delta_t = 0
            max_corr_coef = 0
            corr_coef = 0
            # aligned_df = pd.DataFrame()
            aligned_df = pd.DataFrame(columns=['gmd1_timestamp_t', 'gmd1_t', 'mso_timestamp_t', 'mso_area_t'])
            mso_timestamp_t_copy = df['mso_timestamp_t'].copy()  # 创建 mso_timestamp_t 的副本
            mso_timestamp_t_values = mso_timestamp_t_copy.values
            gmd1_timestamp_t_values = df['gmd1_timestamp_t'].values

            for delta_t in np.arange(-max_delta_t, max_delta_t + step, step):
                aligned_df['gmd1_timestamp_t'] = df['gmd1_timestamp_t']

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

                # print('delta_t:', delta_t, 'corr_coef:', corr_coef)

            # print('best delta_t:', best_delta_t)
            # print('best corr_coef:', max_corr_coef)
            # 保留小数点后10位
            best_delta_t = round(best_delta_t, 10)
            max_corr_coef = round(max_corr_coef, 10)

            self.lineEdit_5.setText(str(best_delta_t))
            self.lineEdit_6.setText(str(max_corr_coef))
            row_shift_pre = 1
            delete_index = []
            max_index = max(df.index)

            for index, gmd1_time in enumerate(gmd1_timestamp_t_values):
                gmd1_time = gmd1_time + best_delta_t
                nearest_index = np.argmin(np.abs(mso_timestamp_t_values - gmd1_time))
                row_shift = nearest_index - index
                if row_shift_pre != row_shift and index - 1 > 0:
                    delete_index.extend([index - 1 + i for i in range(np.abs(row_shift_pre - row_shift) + 1) if index - 1 + i <= max_index])


                aligned_df['gmd1_timestamp_t'].values[index] = gmd1_timestamp_t_values[index]
                aligned_df['gmd1_t'].values[index] = df['gmd1_t'].values[index]
                aligned_df['mso_timestamp_t'].values[index] = df['mso_timestamp_t'].values[nearest_index]
                aligned_df['mso_area_t'].values[index] = df['mso_area_t'].values[nearest_index]
                row_shift_pre = row_shift

            aligned_df['gmd1_timestamp_t'] = aligned_df['gmd1_timestamp_t'] + best_delta_t


            if linear_flag:
                # 计算线性回归模型
                slope, intercept, _, _, _ = linregress(aligned_df['gmd1_t'].tolist(), aligned_df['mso_area_t'].tolist())

                # 计算残差
                residuals = aligned_df['mso_area_t'] - (slope * aligned_df['gmd1_t'] + intercept)

                # 设置拟合残差阈值，超过阈值的行将被删除
                threshold = 1 * np.std(residuals)  # 可根据需要调整阈值

                # 删除拟合残差超过阈值的行
                aligned_df = aligned_df[np.abs(residuals) < threshold]

                # 重新计算相关系数
                corr_coef = pearsonr(aligned_df['gmd1_t'].tolist(), aligned_df['mso_area_t'].tolist())[0]

                # 打印相关系数结果
                # print('Original correlation coefficient:', corr_coef)
                # print(len(aligned_df))
                # 保留小数点后10位
                corr_coef = round(corr_coef, 10)
                self.lineEdit_10.setText(str(corr_coef))
            else:
                aligned_df = aligned_df[~aligned_df.index.isin(delete_index)]
            # 将delta_t写入csv文件
            # with open('data_traverse.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([filename, best_delta_t, corr_coef])
            try:
                # 读取CSV文件为DataFrame
                df_csv = pd.read_csv('data_traverse.csv')
            except FileNotFoundError:
                # 如果不存在csv文件，则创建一个df
                df_csv = pd.DataFrame(columns=['filename', 'delta_t', 'corr_coef'])
                df_csv.to_csv('data_traverse.csv', index=False)
                # 找到与filename匹配的行，并进行删除操作
            df_csv = df_csv[~(df_csv['filename'] == filename)].reset_index(drop=True)
            # 在df_csv中添加一行
            df_csv.loc[len(df_csv)] = [filename, best_delta_t, corr_coef]
            # 将更新后的DataFrame写回文件
            df_csv.to_csv('data_traverse.csv', index=False)

            self.plot(aligned_df)

    def plot_csv(self):
        data = pd.read_csv('delta_t2.csv', header=None, names=['filename', 'delta_t','corr_coef'])
        # pattern = r'/Users/guanzhh/PycharmProjects/tempstamp/timestamp_test/(.*?)/data\.hdf5'
        pattern = r'timestamp_test/(.*?)/data\.hdf5'
        # 提取文件名中的run部分
        x_labels = [re.search(pattern, filename).group(1) for filename in data['filename']]
        # print(x_labels)

        x_ticks = range(len(x_labels))
        self.ax1.cla()
        # 创建图形并绘制数据
        if self.comboBox_3.currentText() == 'Best_delta_t':

            self.ax1.plot(data['delta_t'])
            # 将数据点绘制在折线图上
            self.ax1.scatter(x_ticks, data['delta_t'], color='red')
            self.ax1.set_xticks(x_ticks)
            self.ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            self.ax1.set_xlabel('run')
            self.ax1.set_ylabel('best_delta_t')
            self.ax1.set_title('best_delta_t for each Run')
            self.ax1.grid(True)
            self.ax1.legend(loc='upper right')
            self.canvas1.draw()

        if self.comboBox_3.currentText() == 'Corr_coef':
            self.ax1.plot(data['corr_coef'])
            # 将数据点绘制在折线图上
            self.ax1.scatter(x_ticks, data['corr_coef'], color='red')
            self.ax1.set_xticks(x_ticks)
            self.ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            self.ax1.set_xlabel('run')
            self.ax1.set_ylabel('corr_coef')
            self.ax1.set_title('corr_coef for each Run')
            self.ax1.grid(True)
            self.ax1.legend(loc='upper right')
            self.canvas1.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    display = MyDisplay(parent=window)
    window.setCentralWidget(display)
    window.resize(1600,1100)
    window.show()
    sys.exit(app.exec_())


