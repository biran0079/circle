import numpy as np
from PIL import Image
import pylab
import pickle,sys
import argparse
import os
from matplotlib.widgets import Slider, Button
from scipy.cluster.vq import kmeans
from base import base_name
from scipy.spatial import Delaunay
from scipy.fftpack import fft
import heapq


class PathFinder:
    def __init__(self, fname):
        self.fname = fname
        self.raw_data = pickle.load(open(fname, 'rb'))
        self.k_ratio = 1
        self.sample_ratio = 1

    def _update(self):
        indices = np.random.choice(len(self.raw_data), int(len(self.raw_data) * self.sample_ratio), replace=False)
        self.samples = self.raw_data[indices]
        if self.k_ratio < 1:
            k = int(self.k_ratio * len(self.samples))
            print(f'searching {k} centroid of {len(self.samples)} points')
            self.samples,_ = kmeans(self.samples, k)
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title(f'{str(len(self.samples))} samples')
        self.ax1.scatter(self.samples[:,0], self.samples[:,1], s=0.01)
        print(f'searching mst for {len(self.samples)} points')
        g = self._mst()
        print('searching st, ed')
        st,ed = self._find_farthest_leaf_pair(g)
        print('rearanging children order')
        self._rearange_children_order(g, st, ed)
        self.path = self._generate_path(g, st, ed)
        self.ax2.set_title(f'{int(len(self.path) * 100.0 / len(self.samples) - 100)} % path redundancy')
        self.ax2.plot(self.path[:,0], self.path[:,1])


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
                l,ld,pair,pair_dis = dfs(i, -1)
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
        return st,ed       

    def _generate_path(self, g, st, ed):
        res = []
        vis = set()
        def dfs(i):
            vis.add(i)
            res.append(self.samples[i])
            if i == ed:
                return True
            leaf = True
            for j,_ in g[i]:
                    if j not in vis:
                        leaf = False
                        if dfs(j):
                            return True
            if not leaf:
                # don't visit leaf twice
                res.append(self.samples[i])
            return False
        dfs(st)
        return np.array(res) 

    def _mst(self):
        print('running Delaunay triangulation')
        n=len(self.samples)
        tri = Delaunay(self.samples)
        g = [[] for i in range(n)]

        edges = {}
        nodes = set()
        for simplex in tri.simplices:
            nodes |= set(simplex)
            for k in range(3):
                i, j = simplex[k - 1], simplex[k]
                edge = min(i,j), max(i,j)
                if edge not in edges:
                    edges[edge] = self._dis(i,j)
        pq = [(d, i, j) for ((i,j),d) in edges.items()]
        heapq.heapify(pq)
        p = list(range(n))
        def union(i,j):
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
                union(i,j)
                g[i].append((j,d))
                g[j].append((i,d))
                cc -= 1
        return g

    def save(self):
        X = fft(self.path[...,0] + self.path[...,1] * 1j)
        out_fname = base_name(args.file_name) + ".param"
        print(f'saving to {out_fname}')
        pickle.dump(X, open(out_fname, 'wb'))

    def _update_k_ratio(self, v):
        self.k_ratio = v
        self._update()

    def _update_sample_ratio(self, v):
        self.sample_ratio = v
        self._update()

    def render(self):
        self.fig, (self.ax1, self.ax2) = pylab.subplots(1,2)
        self.fig.set_size_inches([13, 5])
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
    path_finder = PathFinder(args.file_name)
    path_finder.render()
    pylab.show();
    path_finder.save()

if __name__ == '__main__':
    sys.setrecursionlimit(100100)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to contour file')
    args = parser.parse_args()
    main(args)

