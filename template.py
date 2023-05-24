from PyQt5.QtWidgets import QApplication, QMainWindow
from pydm import Display
import sys

class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super(MyDisplay, self).__init__(parent=parent, args=args, macros=macros)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    display = MyDisplay(parent=window)
    window.setCentralWidget(display)
    window.show()
    sys.exit(app.exec_())

# 上面为模版文件
#####


