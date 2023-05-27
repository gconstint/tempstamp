from PyQt5.QtWidgets import QApplication, QMainWindow
from pydm import Display
import sys
from os import path


class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super(MyDisplay, self).__init__(parent=parent, args=args, macros=macros)

    def ui_filename(self):
        return 'gmd_mso.ui'

    def ui_filepath(self):
        return path.join(path.dirname(path.realpath(__file__)), self.ui_filename())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    display = MyDisplay(parent=window)
    window.setCentralWidget(display)
    window.resize(1600, 1100)
    window.show()
    sys.exit(app.exec_())

# 上面为模版文件
#####
