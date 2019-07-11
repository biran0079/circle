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


class Renderer:

    def __init__(self, X, hide, format, n):
        self.frames = len(X)
        coef = np.arange(self.frames)
        if n > 0 and n < self.frames:
            # keep (n + 1)/ 2 first circles and n / 2 last circles
            idx = [i for i in range(self.frames) if i < (
                n + 1) // 2 or self.frames - i <= n // 2]
            X = X[idx]
            coef = coef[idx]
            self.n = n
        else:
            self.n = self.frames

        self.pos = self._find_circle_paths(X, coef)
        self.hide = hide
        self.format = format

    def _find_circle_paths(self, x, coef):
        # pos[i][t] is the position of ith circle's center at t
        # pos[0][t] = 0 for all t
        # pos[i][t] = pos[i - 1][t] + factor[i][t]
        # where factor[i][t] = x[i] * exp(2*pi*t*sqrt(-1)*i/n)
        factor = []
        for i in range(len(coef)):
            factor.append(x[i] * np.exp(2 * np.pi *
                                        1j * coef[i] * np.arange(self.frames) / self.frames) / self.frames)
        pos = [[0 for i in range(self.frames)]]
        for i in range(len(coef)):
            pos.append(pos[-1] + factor[i])
        return pos

    def _init_lines(self):
        if self.hide:
            return
        self.lines = [self.ax.plot([], [], alpha=0.3)[0]
                      for i in range(self.n)]

    def _init_circles(self):
        if self.hide:
            return
        self.circles = []
        for i in range(self.n):
            radius = np.abs(self.pos[i+1][0] - self.pos[i][0])
            circle = plt.Circle((-1, -1), radius, alpha=0.3, fill=False)
            self.circles.append(circle)
            self.ax.add_artist(circle)

    def _init_frame(self):
        self._init_lines()
        self._init_circles()
        self.ax.set(xlim=[0, 1], ylim=[0, 1])
        self.ax.axis('off')
        self.path_x = []
        self.path_y = []
        self.path = self.ax.plot([], [])[0]

    def _render_lines(self, k):
        if self.hide:
            return
        for i in range(self.n):
            p, q = self.pos[i][k], self.pos[i+1][k]
            self.lines[i].set_data([p.real, q.real], [p.imag, q.imag])

    def _render_circles(self, k):
        if self.hide:
            return
        for i in range(self.n):
            p = self.pos[i][k]
            self.circles[i].center = (p.real, p.imag)

    def _render_frame(self, k):
        if k % 100 == 0:
            progress = 100.0 * k / self.frames
            print(f'{progress} %')
        self._render_lines(k)
        self._render_circles(k)
        p = self.pos[self.n][k]
        self.path_x.append(p.real)
        self.path_y.append(p.imag)
        self.path.set_data(self.path_x, self.path_y)

    def render(self):
        print(f'drawing with {self.n} circles')
        self.fig, self.ax = pylab.subplots()
        self.animation = FuncAnimation(self.fig,
                                       self._render_frame,
                                       frames=range(self.frames),
                                       interval=10,
                                       init_func=self._init_frame,
                                       repeat=False)

    def save(self, out_fname):
        if not out_fname:
            out_fname = base_name(args.file_name) + '.' + self.format
        print(f'saving to {out_fname}')
        if self.format == 'mp4':
            self.animation.save(
                out_fname, writer=FFMpegWriter(fps=50, bitrate=1000))
        elif self.format == 'gif':
            self.animation.save(
                out_fname, writer='imagemagick', fps=10, bitrate=100)


def main(args):
    X = pickle.load(open(args.file_name, 'rb'))
    renderer = Renderer(X, args.hide, args.format, args.n)
    renderer.render()
    if args.save:
        renderer.save(args.out)
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to contour file')
    parser.add_argument('--save', default=False,
                        action='store_true', help='save animation')
    parser.add_argument('--hide', default=False,
                        action='store_true', help='hide lines and circles')
    parser.add_argument('--out', default=None, type=str,
                        help='save animation to output file name')
    parser.add_argument('--format', default='mp4', type=str,
                        help='save animation in format')
    parser.add_argument('-n', default=-1, type=int,
                        help='number of circles to use. Not limit by default.')
    args = parser.parse_args()
    main(args)
