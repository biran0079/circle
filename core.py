import numpy as np
import os

def base_name(fname):
    return os.path.splitext(os.path.basename(fname))[0]

def moon(n, c, t):
    return c * np.e ** (n * 1j * t)

def cn(n, f, dt):
    x = np.arange(0, 2 * np.pi, dt)
    return np.sum(moon(-n, f(x), x)) * dt / 2 / np.pi
