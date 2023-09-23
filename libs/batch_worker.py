from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThread
import time


class BatchWorker(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, m_img_list, save_dir, task_function, index = 0):
        super().__init__()
        self.task_function = task_function
        self.m_img_list = m_img_list
        self.save_dir = save_dir
        self.index = index

    def run(self):
        self.progress_signal.emit(0)
        for i, e in enumerate(self.m_img_list):
            self.task_function(e)
            self.progress_signal.emit(i+1)

class ModalDialog(QDialog):
    def __init__(self, parent, m_img_list, save_dir,task_function):
        super().__init__(parent)

        self.setWindowTitle("Batch Emotion Annotator Progress")
        self.setGeometry(100, 100, 450, 150)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, len(m_img_list))
        self.progress_bar.setGeometry(30, 30, 390, 25)

        self.worker = BatchWorker(m_img_list, save_dir, task_function)
        self.worker.progress_signal.connect(self.update_progress)

        self.start_button = QPushButton("Start Task", self)
        self.start_button.setGeometry(30, 80, 100, 30)
        self.start_button.clicked.connect(self.start_task)

        self.close_button = QPushButton("Cancel", self)
        self.close_button.setGeometry(320, 80, 100, 30)
        self.close_button.clicked.connect(self.close)

    def start_task(self):
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if(value == self.progress_bar.maximum()):
            self.close()
