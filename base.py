import numpy as np
import os

def base_name(fname, with_extension=False):
    fname = os.path.basename(fname)
    if not with_extension:
        fname = os.path.splitext(fname)[0]
    return fname
