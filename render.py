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
from matplotlib.widgets import CheckButtons
import time
from functools import partial

class Plotter:

    def __init__(self, X, hide, n):
        self.X = X
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

    def _init_lines(self, ax):
        self.lines = [ax.plot([], [], alpha=0.3)[0]
                      for i in range(self.n)]
    
    def _init_circles(self, ax):
        self.circles = []
        for i in range(self.n):
            radius = np.abs(self.pos[i+1][0] - self.pos[i][0])
            circle = plt.Circle((-1, -1), radius, alpha=0.3, fill=False)
            self.circles.append(circle)
            ax.add_artist(circle)

    def init_frame(self, ax):
        ax.clear()
        ax.set(xlim=[0, 1], ylim=[0, 1])
        ax.axis('off')
        self._init_lines(ax)
        self._init_circles(ax)
        self._update_hide(self.hide)
        self.path_x = []
        self.path_y = []
        self.path = ax.plot([], [])[0]

    def _render_lines(self, k):
        for i in range(self.n):
            p, q = self.pos[i][k], self.pos[i+1][k]
            self.lines[i].set_data([p.real, q.real], [p.imag, q.imag])

    def _render_circles(self, k):
        for i in range(self.n):
            p = self.pos[i][k]
            self.circles[i].center = (p.real, p.imag)

    def render_frame(self, k):
        start_time = time.time()
        self._render_lines(k)
        self._render_circles(k)
        p = self.pos[self.n][k]
        self.path_x.append(p.real)
        self.path_y.append(p.imag)
        self.path.set_data(self.path_x, self.path_y)
        if k % 10 == 0:
            progress = 100.0 * k / self.frames
            elapse = time.time() - start_time
            print(f'{progress} %, {elapse * 1000} ms/frame')

    def _update_hide(self, value):
        self.hide = value
        for i in range(self.n):
            self.lines[i].set_visible(not self.hide)
            self.circles[i].set_visible(not self.hide)

    def toggle_hide(self):
        self._update_hide(not self.hide)


class Renderer:

    def __init__(self, X, hide, n, interval = 100):
        self.plotter = Plotter(X, hide, n)
        self.interval = interval
    
    def render(self):
        print(f'drawing with {self.plotter.n} circles')
        self.fig, self.ax = pylab.subplots()
        pylab.subplots_adjust(bottom = 0.1)
        rax = plt.axes([0.05, 0.0, 0.1, 0.1])
        rax.axis('off')
        self.check = CheckButtons(rax, ('hide',), (self.plotter.hide,))
        self.check.on_clicked(lambda _: self.plotter.toggle_hide())
        
        nax = self.fig.add_axes([0.2, 0.07, 0.7, 0.02])
        self.nslider = Slider(nax, 'n', 2, self.plotter.n,
                           valinit = self.plotter.n, valstep = 1)
        self.nslider.on_changed(self._update_n)
        fpsax = self.fig.add_axes([0.2, 0.03, 0.7, 0.02])
        self.fpsslider = Slider(fpsax, 'fps', 1, 50,
                           valinit = 10, valstep = 1)
        self.fpsslider.on_changed(self._update_fps)
        self._init_animation()

    def _init_animation(self):
        self.animation = FuncAnimation(self.fig,
                                       self.plotter.render_frame,
                                       frames=range(self.plotter.frames),
                                       interval=self.interval,
                                       repeat_delay=1000,
                                       init_func=partial(self.plotter.init_frame, self.ax),
                                       repeat=True)

    def _update_fps(self, fps):
        self.animation.event_source.stop()
        self.interval = int(1000 / fps)
        self._init_animation()

    def _update_n(self, n):
        self.animation.event_source.stop()
        self.plotter = Plotter(self.plotter.X, self.plotter.hide, int(n))
        self._init_animation()

    def save(self, out_fname):
        if not out_fname:
            out_fname = base_name(args.file_name) + '.mp4'
        print(f'saving to {out_fname}')
        self.animation.save(
            out_fname, writer=FFMpegWriter(fps=20, bitrate=1000))

def main(args):
    X = pickle.load(open(args.file_name, 'rb'))
    renderer = Renderer(X, args.hide, args.n)
    pylab.rcParams['toolbar'] = 'None'
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
    parser.add_argument('-n', default=-1, type=int,
                        help='number of circles to use. Not limit by default.')
    args = parser.parse_args()
    main(args)
