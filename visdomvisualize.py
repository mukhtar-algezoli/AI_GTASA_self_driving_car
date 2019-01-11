import numpy 
from visdom import Visdom

class Plot(object):
    def __init__(self, title, port=8097):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel):
        win = self.viz.scatter(
            X=numpy.zeros((1, 2)),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_scatterplot(self, name, x, y):
        self.viz.line(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.windows[name],
            update = 'append'
        )
    def register_line(self, name, xlabel, ylabel):
        win = self.viz.line(
            Y=(numpy.linspace(0, 1, 100000)),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_line(self, name, x, y):
        self.viz.line(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.windows[name],
            update = 'append'
        )    