import numpy as np
from PIL import Image
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
from crdp import rdp

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
        self.sample_ratio = 0.3
        self.smooth_window = 11
        self.epsilon = 0.01
        self.levels = [100 for i in range(level_n)]
        self.contours = None

    def _initialize_raw_data(self):
        self.raw_data = []
        reverse_idx = {}
        self.contour_ax.clear()
        self.contour_ax.axis('off')
        self.contour_ax.set_title(f'{sum(len(seg) for seg in self.contours.allsegs)} contours')
        self.contour_ax.set(xlim=(0, 1), ylim=(0, 1))
        for seg in self.contours.allsegs:
            for poly in seg:
                poly = np.array(poly) / max(self.img.width, self.img.height)
                # down sample 1: make points more discrete
                poly = (np.array(poly) * 1000).astype(int) / 1000
                poly = [tuple(p) for p in poly]

                # index points and deduplicate points shared cross polygons
                poly2 = []
                for p in poly:
                    if p not in reverse_idx:
                        reverse_idx[p] = len(reverse_idx)
                        self.raw_data.append(np.array(p))
                        poly2.append(p)
                self.contour_ax.plot([p[0] for p in poly2], [
                                        p[1] for p in poly2])
       
        self.raw_data = np.array(self.raw_data)

    def _smooth_path(self, path):
        if self.smooth_window == 1:
            return path
        print('smoothing path with savgol_filter')
        before_len = sum(np.linalg.norm(path[i]-path[i-1]) for i in range(1, len(path)))
        from scipy.signal import savgol_filter
        path = np.array(path)
        path[:,0] = savgol_filter(path[:,0], self.smooth_window, 2)
        path[:,1] = savgol_filter(path[:,1], self.smooth_window, 2)
        after_len = sum(np.linalg.norm(path[i]-path[i-1]) for i in range(1, len(path)))
        print(f'path length after smoothing: {before_len} -> {after_len}')
        return path

    def _rdp_downsample(self, path):
        if self.epsilon == 0:
            return path
        print('down sampling path with rdp')
        before_n = len(path)
        path = rdp(path, epsilon=self.epsilon)
        after_n = len(path)
        print(f'samples on path path after rdp: {before_n} -> {after_n}')
        return path

    def _compute_path(self):
        def do_work(self):
            if len(self.samples) <= 1:
                self.path = np.array([])
                return
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
            print(f'max dis on path: {max_dis}')
            if self._dis(st, ed) < 2 * max_dis:
                print('connecting end and start')
                path.append(self.samples[st])
            path = self._smooth_path(path)
            path = self._rdp_downsample(path)
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
            self.imgL, origin='image', levels=sorted(set(self.levels)))
        self._initialize_raw_data()
        # down sample 2: randomly choose points with sample ratio
        indices = np.random.choice(len(self.raw_data), int(
            len(self.raw_data) * self.sample_ratio), replace=False)
        self.samples = self.raw_data[indices]
        self.sample_ax.clear()
        self.path_ax.clear()
        self.image_ax.axis('off')
        self.sample_ax.axis('off')
        self.path_ax.axis('off')
        self.sample_ax.set_title(f'{str(len(self.samples))} samples')
        self.sample_ax.scatter(self.samples[:, 0], self.samples[:, 1], s=0.1)
        self._compute_path()
        self.path_ax.set_title(f'{len(self.path)} points on path')
        if len(self.path):
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
            for j, _ in g[i]:
                if j == parent:
                    continue
                l, ld, pair, pair_dis = dfs(j, i)
                leave_dis.append((ld + 1, l))
                if ld + 1 > farthest_leaf_dis:
                    farthest_leaf_dis = ld + 1
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
        out_fname = f'output/{base_name(self.file_name)}.param'
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

    def _update_epsilon(self, v):
        self.epsilon = v
        self._update()

    def _update_smooth_window(self, v):
        self.smooth_window = int(v)
        self._update()

    def render(self):
        self.fig = Figure(figsize=(13, 13), dpi=100)
        ((self.image_ax, self.contour_ax),  (self.sample_ax,
                                             self.path_ax)) = self.fig.subplots(2, 2)
        self.root.wm_title(f"Compute - {base_name(self.file_name, True)}")
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.image_ax.set_title('image')
        self.image_ax.imshow(self.img)
        self.level_sliders = []
        for i in range(len(self.levels)):
            level = self.fig.add_axes(
                [0.15, 0.015 * (i + 1), 0.7, 0.01])
            slevel = Slider(level, f'contour level {i}', 0, 255, valinit=100)
            slevel.on_changed(self._update_level_fun(i))
            self.level_sliders.append(slevel)
        self.fig.set_size_inches([13, 13])
        ratio = self.fig.add_axes(
            [0.15, 0.015 * (len(self.levels) + 1), 0.7, 0.01])
        self.sratio = Slider(ratio, 'sample ratio', 0.01, 1,
                             valinit=self.sample_ratio, valstep=0.01)
        self.sratio.on_changed(self._update_sample_ratio)
        epsilon = self.fig.add_axes(
            [0.15, 0.015 * (len(self.levels) + 2), 0.7, 0.01])
        self.sepsilon = Slider(epsilon, 'epsilon', 0.00, 0.03,
                             valinit=self.epsilon, valstep=0.001)
        self.sepsilon.on_changed(self._update_epsilon)
        window = self.fig.add_axes(
            [0.15, 0.015 * (len(self.levels) + 3), 0.7, 0.01])
        self.swindow = Slider(window, 'smooth window', 1, 99,
                             valinit=self.smooth_window, valstep=2)
        self.swindow.on_changed(self._update_smooth_window)
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
