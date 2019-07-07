import numpy as np
from PIL import Image
import pylab
import pickle,sys
import argparse
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from core import base_name
from scipy.spatial import Delaunay
import heapq


class PathFinder:
    def __init__(self, fname):
        self.fname = fname
        self.data = pickle.load(open(fname, 'rb'))
        self.data = np.unique(self.data, axis=0)

    def _dis(self, i, j):
        return np.linalg.norm(self.data[i] - self.data[j])

    def _find_st_ed(self, g):
        # makes sure closeset leaf are start and end of dfs
        leaves = [i for i in range(len(g)) if len(g[i]) ==  1]
        print(f'{len(leaves)} leaves in mst')
        dis = squareform(pdist(self.data[leaves]))
        pairs = [(-dis[i][j], leaves[i], leaves[j]) for i in range(len(leaves)) for j in range(i + 1, len(leaves))]
        close_pairs = heapq.nlargest(100, pairs)
        print(f'{len(close_pairs)} close leave pairs in mst')

        vis=set()
        def dfs2(i,ed, res):
            vis.add(i)
            if i == ed: return res
            for j,d in g[i]:
                if j not in vis:
                    t= dfs2(j, ed, res+d)
                    if t is not None:
                        return t
            return None

        def dfs3(st,ed):
            vis.clear()
            return dfs2(st,ed,0)

        print(f'searching for st and ed points in {len(close_pairs)} pairs')
        _,st,ed = max([(dfs3(i,j),i,j) for (_,i,j) in close_pairs])
        return st,ed

    def _rearange_children_order(self, g, st, ed):
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
            res.append(self.data[i])
            if i == ed:
                res.append(self.data[st])
                return True
            leaf = True
            for j,_ in g[i]:
                    if j not in vis:
                        leaf = False
                        if dfs(j):
                            return True
            if not leaf:
                # don't visit leaf twice
                res.append(self.data[i])
            return False
        dfs(st)
        return np.array(res) 

    def _mst(self):
        print('running Delaunay triangulation')
        n=len(self.data)
        tri = Delaunay(self.data)
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

    def _update(self):
        print(f'searching mst for {len(self.data)} points')
        g = self._mst()
        print('searching st, ed')
        st,ed = self._find_st_ed(g)
        print('rearanging children order')
        self._rearange_children_order(g, st, ed)
        self.path = self._generate_path(g, st, ed)
        self.ax.clear()
        self.ax.set_title(f'{len(self.path) * 1.0 / len(self.data)}')
        self.ax.plot(self.path[:,0], self.path[:,1])

    def save(self):
        out_fname = base_name(self.fname) + ".path"
        print(f'saving to {out_fname}')
        pickle.dump(np.array(self.path), open(out_fname, 'wb'))

    def render(self):
        self.fig, self.ax = pylab.subplots()
        self._update()

def main(args):
    path_finder = PathFinder(args.file_name)
    path_finder.render()
    pylab.show();
    path_finder.save()

if __name__ == '__main__':
    import argparse
    import sys
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to sample file')
    args = parser.parse_args()
    main(args)

