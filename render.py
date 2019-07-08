from scipy.interpolate import interp1d
from PIL import Image
import pylab
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pickle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, Button, RadioButtons
from base import base_name
from scipy.fftpack import ifft


class Renderer:
    def __init__(self, X, hide, format):
        self.n = len(X)
        if hide:
            self.ps = ifft(X)
        else:
            self.ps_history = [[0 for i in range(self.n)]] + [ifft(X[:i + 1], n=self.n) for i in range(len(X))]
            self.ps = self.ps_history[-1]
        self.hide = hide
        self.format = format

    def _init_lines(self):
        if self.hide:
            return
        self.lines = [self.ax.plot([],[])[0] for i in range(self.n)]

    def _init_circles(self):
        if self.hide:
            return
        self.circles = [plt.Circle((-1,-1), np.abs(self.ps_history[i+1][0] - self.ps_history[i][0]), fill=False) for i in range(self.n)]
        for circle in self.circles:
            self.ax.add_artist(circle) 

    def _init_frame(self):
        self._init_lines()
        self._init_circles()
        self.ax.set(xlim=[0,1], ylim=[0,1])
        self.ax.axis('off')
        self.path_x = []
        self.path_y = []
        self.path = self.ax.plot([],[])[0]

    def _render_lines(self, k):
        if self.hide:
            return
        for i in range(self.n):
            p,q = self.ps_history[i][k], self.ps_history[i+1][k]
            self.lines[i].set_data([p.real,q.real], [p.imag,q.imag])

    def _render_circles(self, k):
        if self.hide:
            return
        for i in range(self.n):
            p = self.ps_history[i][k]
            self.circles[i].center = (p.real, p.imag)

    def _render_frame(self, k): 
        if k % 100 == 0:
            progress = 100.0 * k / self.n
            print(f'{progress} %')
        self._render_lines(k)
        self._render_circles(k)
        p = self.ps[k]
        self.path_x.append(p.real)
        self.path_y.append(p.imag)
        self.path.set_data(self.path_x, self.path_y)

    def render(self):
        print(f'drawing with {self.n} circles')
        self.fig, self.ax = pylab.subplots()
        axcolor = 'lightgoldenrodyellow'
        self.animation = FuncAnimation(self.fig, 
                self._render_frame,
                frames=range(self.n),
                interval=10,
                init_func=self._init_frame,
                repeat=False)

    def save(self, out_fname):
        if not out_fname:
            out_fname = base_name(args.file_name) + '.' + self.format
        print(f'saving to {out_fname}')
        if self.format == 'mp4':
            self.animation.save(out_fname, writer=FFMpegWriter(fps=50, bitrate=300))
        elif self.format == 'gif':
            self.animation.save(out_fname, writer='imagemagick', fps=10, bitrate=100)


def main(args):
    X = pickle.load(open(args.file_name, 'rb'))
    renderer = Renderer(X, args.hide, args.format)
    renderer.render()
    if args.save:
        renderer.save(args.out)
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to contour file')
    parser.add_argument('--save', default=False, action='store_true', help='save animation')
    parser.add_argument('--hide', default=False, action='store_true', help='hide lines and circles')
    parser.add_argument('--out', default=None, type=str, help='save animation to output file name')
    parser.add_argument('--format', default='mp4', type=str, help='save animation in format')
    args = parser.parse_args()
    main(args)
