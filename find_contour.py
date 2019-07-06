import numpy as np
from PIL import Image
import pylab
import pickle,sys
import argparse
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from core import base_name


def main(args):
    img = Image.open(args.file_name).convert('L')
    contours = [ax.contour(img, origin='image', levels=[80])]
    axcolor = 'lightgoldenrodyellow'
    pylab.subplots_adjust(bottom=0.15)
    level = fig.add_axes([0.15, 0.05, 0.55, 0.04], facecolor=axcolor)
    slevel = Slider(level, 'level', 0, 255, valinit=100)
    save = fig.add_axes([0.8, 0.05, 0.1, 0.04])
    bsave = Button(save, 'save', color=axcolor, hovercolor='0.975')
    def save(_):
        out_fname = base_name(args.file_name) + ".contour"
        print(f'saving to {out_fname}')
        pickle.dump(np.array(contours[0].allsegs[0]), open(out_fname, 'wb'))

    bsave.on_clicked(save)
    def update_level(v):
        for coll in contours.pop().collections:
            coll.remove()
        contours.append(ax.contour(img, origin='image', levels=[v]))

    slevel.on_changed(update_level)
    pylab.show();

if __name__ == '__main__':
    fig, ax = pylab.subplots()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to image file')
    args = parser.parse_args()
    main(args)

