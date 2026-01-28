from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QSize, Qt 

import sys 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("Musializer")
        self.setFixedSize(QSize(800, 600))
        self.setStyleSheet("background-color: #10060A")

        # label 
        self.heading = QLabel("Welcome to the Musializer....")
        self.heading.setStyleSheet("""
            QLabel{
            color: #fff;
            font-size = 16px;
            font-weight:bold;
            }
        """)

        #  layout management

        layout = QVBoxLayout()
        layout.addWidget(self.heading, alignment=Qt.AlignmentFlag.AlignHCenter)

        #container widget to hold the layout 
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

if __name__ == "__main__":
    app = QApplication(sys.argv)


    window = MainWindow()
    window.show()

    app.exec()



