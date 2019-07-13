import numpy as np
from PIL import Image
import pylab
import pickle
import sys
import argparse
import os
from matplotlib.widgets import Slider, Button
from scipy.cluster.vq import kmeans
from base import base_name
from scipy.spatial import Delaunay
from scipy.fftpack import fft
import heapq
import threading
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
from functools import partial

class ParamComputer:
    def __init__(self, file_name, level_n):
        self.root = tkinter.Tk()
        if file_name:
            self.file_name = file_name
        else:
            self.file_name = askopenfilename(title="Select file", filetypes=[
                ("image files", "*.jpg *.gif *.png")])
        self.img = Image.open(self.file_name)
        self.imgL = self.img.convert('L')
        self.sample_ratio = 1
        self.levels = [100 for i in range(level_n)]
        self.contours = None

    def _initialize_raw_data(self):
        res = []
        for seg in self.contours.allsegs:
            for poly in seg:
                for point in poly:
                    res.append(point)
        res = np.array(res)
        res = np.unique(res, axis=0)
        # normalize to fit into into 1 x 1 difure
        res = res / (1.05 * np.max(res))
        self.raw_data = np.array(res)
    
    def _compute_path(self):
        def do_work(self):
            print(f'searching mst for {len(self.samples)} points')
            g = self._mst()
            print('searching st, ed')
            st, ed = self._find_farthest_leaf_pair(g)
            print('rearanging children order')
            self._rearange_children_order(g, st, ed)
            path = self._generate_path(g, st, ed)
            # connect start and end if not too far apart
            max_dis = max([np.linalg.norm(path[i-1] - path[i])
                        for i in range(1, len(path))])
            if self._dis(st, ed) < 2 * max_dis:
                path.append(self.samples[st])
            self.path = np.array(path)
        # Workaround stack size limit on windows.
        # https://stackoverflow.com/questions/2917210/python-what-is-the-hard-recursion-limit-for-linux-mac-and-windows/2918118#2918118
        threading.stack_size(100 * 1024 * 1024)
        max_rec_depth = len(self.samples) + 100
        sys.setrecursionlimit(max_rec_depth)
        thread = threading.Thread(target=partial(do_work, self))
        thread.start()
        thread.join()

    def _update(self):
        if self.contours:
            for coll in self.contours.collections:
                coll.remove()
        self.contours = self.contour_ax.contour(
            self.imgL, origin='image', colors='black', levels=sorted(set(self.levels)))
        self._initialize_raw_data()
        indices = np.random.choice(len(self.raw_data), int(
            len(self.raw_data) * self.sample_ratio), replace=False)
        self.samples = self.raw_data[indices]
        self.sample_ax.clear()
        self.path_ax.clear()
        self.image_ax.axis('off')
        self.contour_ax.axis('off')
        self.sample_ax.axis('off')
        self.path_ax.axis('off')
        self.sample_ax.set_title(f'{str(len(self.samples))} samples')
        self.sample_ax.scatter(self.samples[:, 0], self.samples[:, 1], s=0.1)
        self._compute_path()
        self.path_ax.set_title(
            f'{int(len(self.path) * 100.0 / len(self.samples) - 100)} % path redundancy')
        self.path_ax.plot(self.path[:, 0], self.path[:, 1], alpha=0.5)

    def _dis(self, i, j):
        return np.linalg.norm(self.samples[i] - self.samples[j])

    def _find_farthest_leaf_pair(self, g):
        def dfs(i, parent):
            """
            Return
                - farthest leaf id in thissubtree and distance to root i
                - farthest leave pair in this subtree and distance between them
            """
            farthest_leaf = i
            farthest_leaf_dis = 0
            farthest_leaf_pair = None
            farthest_leaf_pair_dis = -1
            leave_dis = []
            for j, d in g[i]:
                if j == parent:
                    continue
                l, ld, pair, pair_dis = dfs(j, i)
                leave_dis.append((ld + d, l))
                if ld + d > farthest_leaf_dis:
                    farthest_leaf_dis = ld + d
                    farthest_leaf = l
                if farthest_leaf_pair_dis < pair_dis:
                    farthest_leaf_pair = pair
                    farthest_leaf_pair_dis = pair_dis
            if len(leave_dis) >= 2:
                (d1, l1), (d2, l2) = sorted(leave_dis)[-2:]
                if d1 + d2 > farthest_leaf_pair_dis:
                    farthest_leaf_pair_dis = d1 + d2
                    farthest_leaf_pair = l1, l2
            return farthest_leaf, farthest_leaf_dis, farthest_leaf_pair, farthest_leaf_pair_dis

        for i in range(len(g)):
            if len(g[i]):
                l, ld, pair, pair_dis = dfs(i, -1)
                if len(g[i]) == 1 and ld > pair_dis:
                    # root is a leave
                    return i, l
                return pair

    def _rearange_children_order(self, g, st, ed):
        # reagange children list order to make sure ed is the last node to visit
        # when starting from st
        vis = set()

        def dfs(i):
            vis.add(i)
            if i == ed:
                return True
            for j in range(len(g[i])):
                if g[i][j][0] not in vis:
                    if dfs(g[i][j][0]):
                        g[i][j], g[i][-1] = g[i][-1], g[i][j]
                        return True
            return False
        dfs(st)
        return st, ed

    def _generate_path(self, g, st, ed):
        res = []
        vis = set()

        def dfs(i):
            vis.add(i)
            res.append(self.samples[i])
            if i == ed:
                return True
            leaf = True
            for j, _ in g[i]:
                if j not in vis:
                    leaf = False
                    if dfs(j):
                        return True
            if not leaf:
                # don't visit leaf twice
                res.append(self.samples[i])
            return False
        dfs(st)
        return res

    def _mst(self):
        print('running Delaunay triangulation')
        n = len(self.samples)
        tri = Delaunay(self.samples)
        g = [[] for i in range(n)]

        edges = {}
        nodes = set()
        for simplex in tri.simplices:
            nodes |= set(simplex)
            for k in range(3):
                i, j = simplex[k - 1], simplex[k]
                edge = min(i, j), max(i, j)
                if edge not in edges:
                    edges[edge] = self._dis(i, j)
        pq = [(d, i, j) for ((i, j), d) in edges.items()]
        heapq.heapify(pq)
        p = list(range(n))

        def union(i, j):
            p[find(i)] = find(j)

        def find(i):
            if p[i] == i:
                return i
            p[i] = find(p[i])
            return p[i]
        print('running kruskal')
        # nodes may not contain all points as some points close to each other are treated as single points
        cc = len(nodes)
        while cc > 1:
            d, i, j = heapq.heappop(pq)
            if find(i) != find(j):
                union(i, j)
                g[i].append((j, d))
                g[j].append((i, d))
                cc -= 1
        return g

    def save(self):
        X = fft(self.path[..., 0] + self.path[..., 1] * 1j)
        out_fname = base_name(self.file_name) + ".param"
        print(f'saving to {out_fname}')
        pickle.dump(X, open(out_fname, 'wb'))

    def _update_sample_ratio(self, v):
        self.sample_ratio = v
        self._update()

    def _update_level_fun(self, i):
        def f(v):
            self.levels[i] = v
            self._update()
        return f

    def render(self):
        self.fig = Figure(figsize=(13, 13), dpi=100)
        ((self.image_ax, self.contour_ax),  (self.sample_ax,
                                             self.path_ax)) = self.fig.subplots(2, 2)
        self.root.wm_title(f"Compute - {base_name(self.file_name, True)}")
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.image_ax.set_title('image')
        self.contour_ax.set_title('contour')
        self.image_ax.imshow(self.img)
        self.level_sliders = []
        for i in range(len(self.levels)):
            level = self.fig.add_axes(
                [0.15, 0.015 * (i + 1), 0.7, 0.01])
            slevel = Slider(level, f'level {i}', 0, 255, valinit=100)
            slevel.on_changed(self._update_level_fun(i))
            self.level_sliders.append(slevel)
        self.fig.set_size_inches([13, 13])
        ratio = self.fig.add_axes(
            [0.15, 0.015 * (len(self.levels) + 1), 0.7, 0.01])
        self.sratio = Slider(ratio, 'sample ratio', 0.01, 1,
                             valinit=1, valstep=0.01)
        self.sratio.on_changed(self._update_sample_ratio)
        self._update()
        tkinter.mainloop()


def main(args):
    param_computer = ParamComputer(args.file_name, args.level_n)
    param_computer.render()
    param_computer.save()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str,
                        help='path to contour file', default='')
    parser.add_argument('-n', '--level_n', type=int,
                        help='number of levels', default=1)
    args = parser.parse_args()
    main(args)
