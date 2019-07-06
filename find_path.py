import numpy as np
from PIL import Image
import pylab
import pickle,sys
import argparse
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.cluster.vq import kmeans
from core import base_name

def mst(a):
    n=len(a)
    p = list(range(n))
    def union(i,j):
       p[find(i)] = find(j)
    def find(i):
       if p[i] == i:
           return i
       p[i] = find(p[i])
       return p[i]
    dis = np.linalg.norm(a.reshape(1,n,2) - a.reshape(n,1,2), axis=2)
    import heapq
    pq = [(dis[i][j], i, j) for i in range(n) for j in range(i + 1, n)]
    heapq.heapify(pq)
    cc = n
    g = [[] for i in range(n)]
    while cc > 1:
      _, i, j = heapq.heappop(pq)
      if find(i) != find(j):
          union(i,j)
          g[i].append(j)
          g[j].append(i)
          cc -= 1
    return dis, g

def rearange_leave_order(dis, g):
    # makes sure closeset leaf are start and end of dfs
    leaves = [i for i in range(len(g)) if len(g[i]) == 1]
    pairs = [(leaves[i], leaves[j]) for i in range(len(leaves)) for j in range(i + 1, len(leaves))]
    st, ed = min(pairs, key=lambda x:dis[x[0]][x[1]])
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

def generate_path(centroids, g, st, ed):
    res = []
    vis = set()
    def dfs(i):
       vis.add(i)
       res.append(centroids[i])
       if i == ed:
           res.append(centroids[st])
           return True
       leaf = True
       for j in g[i]:
            if j not in vis:
                leaf = False
                if dfs(j):
                    return True
       if not leaf:
           # don't visit leaf twice
           res.append(centroids[i])
       return False
    dfs(st)
    return np.array(res) 


def main(args):
    contours = pickle.load(open(args.file_name, 'rb'))
    data = np.concatenate(contours, axis=0)
    path = []
    axcolor = 'lightgoldenrodyellow'
    pylab.subplots_adjust(bottom=0.15)
    ratio = fig.add_axes([0.15, 0.07, 0.55, 0.02], facecolor=axcolor)
    kratio = fig.add_axes([0.15, 0.03, 0.55, 0.02], facecolor=axcolor)
    sratio = Slider(ratio, 'ratio', 1, 20, valinit=10, valstep=1)
    skratio = Slider(kratio, 'k-ratio', 0.05, 0.95, valinit=0.2, valstep=0.01)
    save = fig.add_axes([0.8, 0.05, 0.1, 0.04])
    bsave = Button(save, 'save', color=axcolor, hovercolor='0.975')
    def save(_):
        out_fname = base_name(args.file_name) + ".path"
        print(f'saving to {out_fname}')
        pickle.dump(np.array(path[0]), open(out_fname, 'wb'))

    bsave.on_clicked(save)
    params = [10,0.2]
    def update():
        ratio,k = params
        t = data[::ratio]
        k = (int)(k * len(t))
        print(f'searching {k} centroid of {len(t)} points')
        centroids,_ = kmeans(t, k)
        print(f'searching mst for {len(t)} centroids')
        dis,g = mst(centroids)
        st,ed = rearange_leave_order(dis, g)
        print(f'start: {centroids[st]}, end: {centroids[ed]}')
        t = generate_path(centroids, g, st, ed)
        print(f'path: {t}')
        path.clear()
        path.append(t)
        ax.clear()
        ax.set_title(str(len(t)))
        ax.plot(t[:,0], t[:,1])

    def update_ratio(v):
        params[0] = int(v)
        update()

    sratio.on_changed(update_ratio)
    def update_k(v):
        params[1] = v
        update()

    skratio.on_changed(update_k)
    update()

    pylab.show();

if __name__ == '__main__':
    fig, ax = pylab.subplots()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='path to contour file')
    args = parser.parse_args()
    main(args)

