import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore

class LiveLossPlot:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)

        self.window = pg.GraphicsLayoutWidget(
            show=True,
            title="MLP Training - MSE"
        )
        self.window.resize(900, 500)

        self.plot = self.window.addPlot(
            title="Mean Squared Error"
        )

        self.plot.setLabel("left", "MSE")
        self.plot.setLabel("bottom", "Epoch")
        self.plot.showGrid(x=True, y=True)

        self.curve = self.plot.plot(
            pen=pg.mkPen("cyan", width=2)
        )

        self.epochs = []
        self.losses = []

    def update(self, epoch, mse):
        self.epochs.append(epoch)
        self.losses.append(mse)

        self.curve.setData(
            self.epochs,
            self.losses
        )

        self.app.processEvents()

    def close(self):
        self.window.close()


plotter = LiveLossPlot()

for epoch in range(1000):
    mse = np.random.normal(loc=3, scale=1)
    plotter.update(epoch, mse)

plotter.app.exec()
#plotter.close()
