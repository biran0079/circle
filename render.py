from scipy.interpolate import interp1d
from PIL import Image
import pylab
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pickle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, Button, RadioButtons
from core import moon,cn, base_name



class Renderer:
    def __init__(self, N, C, frame, hide):
        # sort N in order 0, 1, -1, 2, -2 ...
        nc = np.array(sorted(zip(N,C), key=lambda x:abs(x[0])))
        self.N = nc[:,0]
        self.C = nc[:,1]
        self.frame = frame
        self.hide = hide

    def _init_lines(self):
        if self.hide:
            return
        self.lines = [self.ax.plot([],[])[0] for i in range(len(self.C))]

    def _init_circles(self):
        if self.hide:
            return
        ps = moon(self.N, self.C, 0)
        self.circles = [plt.Circle((-1,-1), np.abs(ps[i]), fill=False) for i in range(len(self.C))]
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

    def _render_lines(self, ps):
        if self.hide:
            return
        p = 0j 
        for idx in range(len(ps)):
            q=ps[idx]
            self.lines[idx].set_data([p.real,q.real], [p.imag,q.imag])
            p = q

    def _render_circles(self, ps):
        if self.hide:
            return
        p=0j 
        for idx in range(len(ps)):
            self.circles[idx].center = (p.real, p.imag)
            p = ps[idx]

    def _render_frame(self, t): 
        progress = 100 * t / 2 / np.pi
        print(f'{progress} %')
        ps = np.cumsum(moon(self.N, self.C, t))
        self._render_lines(ps)
        self._render_circles(ps)
        p = ps[-1]
        self.path_x.append(p.real)
        self.path_y.append(p.imag)
        self.path.set_data(self.path_x, self.path_y)

    def render(self):
        self.fig, self.ax = pylab.subplots()
        axcolor = 'lightgoldenrodyellow'
        self.animation = FuncAnimation(self.fig, 
                self._render_frame,
                frames=np.linspace(0, 2 * np.pi, self.frame),
                interval=10,
                init_func=self._init_frame,
                repeat=False)

    def save(self, out_fname):
        if not out_fname:
            out_fname = base_name(args.file_name) + '.mp4'
        print(f'saving to {out_fname}')
        self.animation.save(out_fname, writer=FFMpegWriter(fps=50, bitrate=300))


def main(args):
    N,C = pickle.load(open(args.file_name, 'rb'))
    renderer = Renderer(N, C, args.frame, args.hide)
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
    parser.add_argument('--frame', type=int, default=500, help='number of frames to finish the plot.')
    parser.add_argument('--out', default=None, type=str, help='save animation to output file name')
    args = parser.parse_args()
    main(args)
