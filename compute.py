from scipy.interpolate import interp1d
from PIL import Image
import pylab
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pickle
import os
from core import moon,cn,base_name


def main(args):
    figure = plt.figure()
    a = figure.subplots()
    path = pickle.load(open(args.file_name, 'rb'))
    path = path[...,0] + path[...,1] * 1j
    f = interp1d(np.linspace(0, 2 * np.pi, len(path)), path)
    circleN = args.num_circle_pairs
    dt = args.dt
    N = np.array(range(-circleN, circleN + 1))
    C = []
    for i in range(len(N)):
        C.append(cn(N[i], f, dt))
        if i % 100 == 0:
            print(f'{i * 100.0 / len(N)} %')
    C = np.array(C)
    out_fname = base_name(args.file_name) + ".param"
    print(f'saving to {out_fname}')
    pickle.dump((N,C), open(out_fname, 'wb'))

if __name__ == '__main__':
    fig, ax = pylab.subplots()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to path file')
    parser.add_argument('--num_circle_pairs', type=int, help='number of circle pairs', default=500)
    parser.add_argument('--dt', type=float, help='numerical integral steps', default=0.0001)
    args = parser.parse_args()
    main(args)
