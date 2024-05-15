import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QSizePolicy, QFileDialog, QLabel, QMessageBox
from PyQt5 import QtMultimedia
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import librosa
import matplotlib

import librosa.display

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from analyze_model import gen_gif, predict
matplotlib.use('Qt5Agg')

class Graph(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot()

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def plotWave(self, filename):
        self.axes.clear()
        data, sr = librosa.load(filename)
        # librosa.display.waveplot(data, sr=sr, ax=self.axes)
        librosa.display.waveplot(data, sr=sr, ax=self.axes)
        f = filename.split("/")
        self.axes.set_title(f[len(f) - 1])
        self.draw()

class InitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fname = ""
        self.initUI()

    def initUI(self):
        self.load_btn = QPushButton("Загрузить аудио", self)
        self.load_btn.setGeometry(QRect(9, 9, 211, 23))
        self.load_btn.clicked.connect(self.btn_load_clicked)

        self.analyze_btn = QPushButton("Анализировать аудио", self)
        self.analyze_btn.setGeometry(QRect(444, 9, 211, 23))
        self.analyze_btn.clicked.connect(self.analyze)

        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(QIcon("icons/playicon.png"))
        self.play_btn.setGeometry(QRect(250, 410, 51, 41))
        self.play_btn.clicked.connect(self.play)

        self.stop_btn = QPushButton(self)
        self.stop_btn.setIcon(QIcon("icons/stopicon.png"))
        self.stop_btn.setGeometry(QRect(300, 410, 51, 41))
        self.stop_btn.clicked.connect(self.stop)

        self.pause_btn = QPushButton(self)
        self.pause_btn.setIcon(QIcon("icons/pauseicon.png"))
        self.pause_btn.setGeometry(QRect(350, 410, 51, 41))
        self.pause_btn.clicked.connect(self.pause)

        self.player = QtMultimedia.QMediaPlayer()

        self.graphic = Graph(self)
        self.graphic.setGeometry(QRect(9, 38, 646, 369))

        self.setFixedSize(664, 463)
        self.setWindowTitle('UrbanSoundsAnalyzer')
        self.setWindowIcon(QIcon('icons/icon.png'))
        self.show()

    def btn_load_clicked(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open File',
                                                 'c:\\', 'Wav files (*.wav)')[0]
        if self.fname != "":
            self.graphic.plotWave(self.fname)
            self.url = QUrl.fromLocalFile(self.fname)
            self.audio = QtMultimedia.QMediaContent(self.url)
            self.player.setMedia(self.audio)

    def play(self):
        self.player.play()

    def stop(self):
        self.player.stop()

    def pause(self):
        self.player.pause()

    def analyze(self):
        if self.fname != "":
            self.analyze_dialog = AnalyzeWindow(self.fname)
            self.analyze_dialog.show()

class AnalyzeWindow(QWidget):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Result')
        self.setFixedSize(700, 500)

        self.label = QLabel(self)
        #self.label.setGeometry(QRect(9, 38, 646, 369))

        self.movie = QMovie(gen_gif(self.fname))
        self.label.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()

        self.frame_count = self.movie.frameCount()
        self.movie.frameChanged.connect(self.frame_changed)

        self.player = QtMultimedia.QMediaPlayer()
        self.url = QUrl.fromLocalFile(self.fname)
        self.audio = QtMultimedia.QMediaContent(self.url)
        self.player.setMedia(self.audio)

        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(QIcon("icons/playicon.png"))
        self.play_btn.setGeometry(QRect(300, 410, 51, 41))
        self.play_btn.clicked.connect(self.play)

        self.stop_btn = QPushButton(self)
        self.stop_btn.setIcon(QIcon("icons/stopicon.png"))
        self.stop_btn.setGeometry(QRect(350, 410, 51, 41))
        self.stop_btn.clicked.connect(self.stop)

        self.msg = QMessageBox(self)
        self.msg.setText(predict(self.fname))
        self.msg.show()

    def frame_changed(self, v):
        if self.frame_count == v + 1:
            self.movie.stop()

    def play(self):
        self.player.play()
        self.movie.start()

    def stop(self):
        self.player.stop()
        self.movie.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    obj = InitWindow()
    sys.exit(app.exec_())
