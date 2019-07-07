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



class PathFinder:
    def __init__(self, fname):
        self.fname = fname
        self.data = pickle.load(open(fname, 'rb'))

    def _find_st_ed(self, dis, g):
        # makes sure closeset leaf are start and end of dfs
        leaves = [i for i in range(len(g)) if len(g[i]) % 2 ==  1]
        pairs = [(leaves[i], leaves[j]) for i in range(len(leaves)) for j in range(i + 1, len(leaves))]
        vis=set()
        def dfs2(i,ed, res):
            vis.add(i)
            if i == ed: return res
            for j in g[i]:
                if j not in vis:
                    t= dfs2(j, ed, res+dis[i][j])
                    if t is not None:
                        return t
            return None

        def dfs3(st,ed):
            vis.clear()
            return dfs2(st,ed,0)

        close_pairs = [(i,j) for (i,j) in pairs if dis[i][j] < 0.05]
        if close_pairs:
            if len(close_pairs) > 100:
                close_pairs = sorted(close_pairs, key=lambda x:dis[x[0]][x[1]])[:100]

            print(f'searching for st and ed points in {len(close_pairs)} pairs')
            _,st,ed = max([(dfs3(i,j),i,j) for (i,j) in close_pairs])
        else:
            st, ed = min(pairs, key=lambda x:dis[x[0]][x[1]])
        return st,ed

    def _rearange_children_order(self, g, st, ed):
        vis = set()
        def dfs(i):
            vis.add(i)
            if i == ed:
                return True
            for j in range(len(g[i])):
                if g[i][j] not in vis:
                    if dfs(g[i][j]):
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
            for j in g[i]:
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

    def _mst(self,a):
        n=len(a)
        g = [[] for i in range(n)]
        print('building distance matrix')
        dis = squareform(pdist(a))
        p = list(range(n))
        def union(i,j):
            p[find(i)] = find(j)
        def find(i):
            if p[i] == i:
                return i
            p[i] = find(p[i])
            return p[i]
        print('running kruskal')
        import heapq
        pq = [(dis[i][j], i, j) for i in range(n) for j in range(i + 1, n)]
        heapq.heapify(pq)
        cc = n
        while cc > 1:
            _, i, j = heapq.heappop(pq)
            if find(i) != find(j):
                union(i,j)
                g[i].append(j)
                g[j].append(i)
                cc -= 1
        return dis, g

    def _update(self):
        print(f'searching mst for {len(self.data)} points')
        dis,g = self._mst(self.data)
        print('searching st, ed')
        st,ed = self._find_st_ed(dis, g)
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

