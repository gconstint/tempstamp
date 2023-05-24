"""
以下代码为设计顶部菜单栏的初步尝试
"""
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction
from pydm import Display
import sys

class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super(MyDisplay, self).__init__(parent=parent, args=args, macros=macros)

        # 在顶部工具栏中添加下拉选项
        menu_bar = self.parent().menuBar()
        menu = menu_bar.addMenu("Options")

        action1 = QAction("Option 1", self)
        action1.triggered.connect(self.option1_selected)
        menu.addAction(action1)

        action2 = QAction("Option 2", self)
        action2.triggered.connect(self.option2_selected)
        menu.addAction(action2)

    def option1_selected(self):
        print("Option 1 selected")

    def option2_selected(self):
        print("Option 2 selected")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    display = MyDisplay(parent=window)
    window.setCentralWidget(display)
    window.show()
    sys.exit(app.exec_())