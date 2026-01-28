from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QSize, Qt 

import sys 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("Musializer")
        self.setFixedSize(QSize(800, 600))
        self.setStyleSheet("background-color: #10060A")


app = QApplication(sys.argv)


window = MainWindow()
window.show()

app.exec()



