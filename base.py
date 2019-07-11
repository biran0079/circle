import numpy as np
import os


def base_name(fname):
    return os.path.splitext(os.path.basename(fname))[0]
