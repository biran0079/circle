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



def init():
    ax.clear()
    global lines,path, path_x, path_y,circles
    lines = [ax.plot([],[])[0] for i in range(len(C))]
    ps = moon(N,C,0)
    circles = [plt.Circle((-1,-1), np.abs(ps[i]), fill=False) for i in range(len(C))]
    for circle in circles:
        ax.add_artist(circle) 
    ax.set(xlim=[0,1], ylim=[0,1])
    ax.axis('off')
    path_x = []
    path_y = []
    path = ax.plot([],[])[0]

def ani(t): 
    print(t)
    ps=moon(N,C,t)
    p=0j 
    for idx in range(len(C)):
        q=ps[idx]+p
        circles[idx].center = (p.real, p.imag)
        lines[idx].set_data([p.real,q.real], [p.imag,q.imag])
        p=q
    path_x.append(p.real)
    path_y.append(p.imag)
    path.set_data(path_x, path_y)


def main(args):
    global N,C
    N,C = pickle.load(open(args.file_name, 'rb'))
    nc = np.array(sorted(zip(N,C), key=lambda x:abs(x[0])))
    N = nc[:,0]
    C = nc[:,1]

    axcolor = 'lightgoldenrodyellow'

    animation = FuncAnimation(fig, ani,frames=np.linspace(0, 2 * np.pi, args.frame), interval=10, init_func=init, repeat=False)
    if args.save:
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Ran Bi'), bitrate=1800)
        out_fname = base_name(args.file_name) + ".mp4"
        print(f'saving to {out_fname}')
        animation.save(out_fname, writer=writer)
    plt.show()

if __name__ == '__main__':
    fig, ax = pylab.subplots()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to contour file')
    parser.add_argument('--save', default=False, action='store_true', help='save animation')
    parser.add_argument('--frame', type=int, default=500, help='number of frames to finish the plot.')
    args = parser.parse_args()
    main(args)
