import numpy as np
from PIL import Image
import pylab
import pickle
import sys
import argparse
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from base import base_name


class ContourFinder:

    def __init__(self, fname, level_n):
        self.fname = fname
        self.img = Image.open(self.fname).convert('L')
        self.levels = [100 for i in range(level_n)]
        self.contours = None

    def _update(self):
        if self.contours:
            for coll in self.contours.collections:
                coll.remove()
        self.contours = self.ax.contour(
            self.img, origin='image', levels=sorted(set(self.levels)))

    def _update_level_fun(self, i):
        def f(v):
            self.levels[i] = v
            self._update()
        return f

    def render(self):
        self.fig, self.ax = pylab.subplots()
        axcolor = 'lightgoldenrodyellow'
        pylab.subplots_adjust(bottom=0.15 * len(self.levels))
        self.sliders = []
        for i in range(len(self.levels)):
            level = self.fig.add_axes(
                [0.15, 0.04 * (i + 1), 0.60, 0.03], facecolor=axcolor)
            slevel = Slider(level, f'level {i}', 0, 255, valinit=100)
            slevel.on_changed(self._update_level_fun(i))
            self.sliders.append(slevel)
        self._update()

    def save(self):
        out_fname = base_name(self.fname) + ".contour"
        print(f'saving to {out_fname}')
        res = []
        for seg in self.contours.allsegs:
            for poly in seg:
                for point in poly:
                    res.append(point)
        res = np.array(res)
        res = np.unique(res, axis=0)
        res = res / (1.05 * np.max(res))
        pickle.dump(res, open(out_fname, 'wb'))


def main(args):
    contour_finder = ContourFinder(args.file_name, args.level_n)
    contour_finder.render()
    pylab.show()
    contour_finder.save()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to image file')
    parser.add_argument('--level_n', type=int,
                        help='number of levels', default=1)
    args = parser.parse_args()
    main(args)
