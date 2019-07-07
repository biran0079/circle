import numpy as np
from PIL import Image
import pylab
import pickle,sys
import argparse
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.cluster.vq import kmeans
from core import base_name



class DownSampler:
    def __init__(self, fname):
        self.fname = fname
        self.data = pickle.load(open(fname, 'rb'))
        self.k_ratio = 1
        self.sample_ratio = 1

    def _update(self):
        samples = np.random.choice(len(self.data), int(len(self.data) * self.sample_ratio), replace=False)
        samples = [self.data[i] for i in samples]
        if self.k_ratio < 1:
            k = int(self.k_ratio * len(samples))
            print(f'searching {k} centroid of {len(samples)} points')
            self.samples,_ = kmeans(samples, k)
        else:
            self.samples = np.array(samples)
        self.ax.clear()
        self.ax.set_title(str(len(self.samples)))
        self.ax.scatter(self.samples[:,0], self.samples[:,1], s=0.01)

    def save(self):
        out_fname = base_name(self.fname) + ".sample"
        print(f'saving to {out_fname}')
        pickle.dump(np.array(self.samples), open(out_fname, 'wb'))

    def _update_k_ratio(self, v):
        self.k_ratio = v
        self._update()

    def _update_sample_ratio(self, v):
        self.sample_ratio = v
        self._update()

    def render(self):
        self.fig, self.ax = pylab.subplots()
        axcolor = 'lightgoldenrodyellow'
        pylab.subplots_adjust(bottom=0.15)
        ratio = self.fig.add_axes([0.15, 0.07, 0.55, 0.02], facecolor=axcolor)
        kratio = self.fig.add_axes([0.15, 0.03, 0.55, 0.02], facecolor=axcolor)
        self.sratio = Slider(ratio, 'ratio', 0.05, 1, valinit=1, valstep=0.05)
        self.skratio = Slider(kratio, 'k-ratio', 0.05, 1, valinit=1, valstep=0.05)
        self.skratio.on_changed(self._update_k_ratio)
        self.sratio.on_changed(self._update_sample_ratio)
        self._update()

def main(args):
    down_sampler  = DownSampler(args.file_name)
    down_sampler.render()
    pylab.show();
    down_sampler.save()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to contour file')
    args = parser.parse_args()
    main(args)

