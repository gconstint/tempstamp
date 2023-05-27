#-*- coding: utf-8 -*-
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout,QVBoxLayout
from pydm import Display
import sys,os
from os import path
import h5py
import numpy as np
import pandas as pd


from scipy.stats import pearsonr
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super(MyDisplay, self).__init__(parent=parent, args=args, macros=macros)

        self.fig1 = Figure(facecolor=(1, 1, 1, 1))
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor((255 / 255, 255 / 255, 255 / 255, 255 / 255))
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)

        self.ax1.set_title('Fig1')
        self.fig1.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.2, wspace=0.1)
        self.canvas1 = FigureCanvas(self.fig1)
        # self.toolbar1 = NavigationToolbar(self.canvas1, self)
        # # add Canvas to Layout
        self.ui.PyDMEmbeddedDisplay_1.layout.addWidget(self.canvas1)
        # self.ui.PyDMEmbeddedDisplay_1.layout.addWidget(self.toolbar1)

        self.fig2 = Figure(facecolor=(1, 1, 1, 1), figsize=(5, 4), dpi=100)
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



        self.fig2.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.1, wspace=0)
        self.canvas2 = FigureCanvas(self.fig2)
        # self.toolbar2 = NavigationToolbar(self.canvas2, self)
        # add Canvas to Layout
        self.ui.PyDMEmbeddedDisplay_2.layout.addWidget(self.canvas2)
        # self.ui.PyDMEmbeddedDisplay_2.layout.addWidget(self.toolbar2)


        self.pushButton_3.clicked.connect(self.collection_2)

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
        file_path, _ = QFileDialog.getOpenFileName(None, "Select file", "/", options=options)

        # If the user selects a file, set its path to the text of the text box
        if file_path:
            self.filename = file_path
            file_name = os.path.basename(file_path)
            self.textbox_1.setText(file_name)

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
        self.ax1.set_title('GMD1 and MSO_AREA over time')
        self.ax1.legend()

        self.ax2.plot(np.diff(aligned_df['gmd1_timestamp_t']), label="gmd.timestamp")
        self.ax2.plot(np.diff(aligned_df['mso_timestamp_t']), label="pd.timestamp")
        self.ax2.set_xlabel("pulse number")
        self.ax2.set_ylabel("interval")
        self.ax2.set_title("interval of timestamp")
        self.ax2.legend()

        self.ax3.plot(aligned_df['mso_timestamp_t'] - aligned_df['gmd1_timestamp_t'], label="timestamp difference")
        self.ax3.set_xlabel("pulse number")
        self.ax3.set_ylabel("difference")
        self.ax3.set_title("difference between %s and %s" % ("pd", "gmd"))
        self.ax3.legend()

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
        self.ax4.legend()

        self.canvas1.draw()
        self.canvas2.draw()
    def collection_1(self):
        pass

    def collection_2(self):
        filename = self.filename
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

                # print('delta_t:', delta_t, 'corr_coef:', corr_coef)

            # print('best delta_t:', best_delta_t)
            # print('best corr_coef:', max_corr_coef)
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

                if np.min(np.abs(mso_timestamp_t_values - gmd1_time)) < 0.05:
                    aligned_df['gmd1_timestamp_t'].values[index] = gmd1_timestamp_t_values[index]
                    aligned_df['gmd1_t'].values[index] = df['gmd1_t'].values[index]
                    aligned_df['mso_timestamp_t'].values[index] = df['mso_timestamp_t'].values[nearest_index]
                    aligned_df['mso_area_t'].values[index] = df['mso_area_t'].values[nearest_index]
                    row_shift_pre = row_shift

            aligned_df['gmd1_timestamp_t'] = aligned_df['gmd1_timestamp_t'] + best_delta_t
            aligned_df = aligned_df[~aligned_df.index.isin(delete_index)]

            self.plot(aligned_df)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    display = MyDisplay(parent=window)
    window.setCentralWidget(display)
    window.resize(1600,1100)
    window.show()
    sys.exit(app.exec_())

# 上面为模版文件
#####
