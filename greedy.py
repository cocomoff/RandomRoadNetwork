# -*- coding: utf-8 -*-

import click
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def unique_generator(G, num, th, used=[]):
    lPass = np.zeros((num, 2), dtype=np.int)
    for p in range(num):
        while True:
            s = np.random.choice(G.nodes)
            while s in used:
                s = np.random.choice(G.nodes)
            t = np.random.choice(G.nodes)
            while s == t or t in used:
                t = np.random.choice(G.nodes)
            dst = nx.shortest_path_length(G, s, t, weight='weight')
            if dst > th:
                lPass[p] = [s, t]
                used.append(s)
                used.append(t)
                break   
    return lPass, used

def plot_stpath(ax, pos, stpath, color='k', lw=2, alpha=0.5):
    for i in range(len(stpath) - 1):
        ii = stpath[i]
        jj = stpath[i + 1]
        posI = pos[ii, :]
        posJ = pos[jj, :]
        ax.plot([posI[0], posJ[0]], [posI[1], posJ[1]], color=color, lw=lw, alpha=alpha)


# greedy passenger insertion at **edge segment**
def greedy(G, lV, lPass):
    routeV, routeP = {}, {}
    nv, npass = lV.shape[0], lPass.shape[0]

    for v in range(nv):
        routeV[v] = nx.shortest_path(G, lV[v, 0], lV[v, 1], weight='weight')

    for p in range(npass):
        sp, ep = lPass[p, 0], lPass[p, 1]
        stp = nx.shortest_path(G, sp, ep, weight='weight')
        mmv, mmiv, mmc = None, None, float("Inf")
        for v in range(nv):
            rv = routeV[v]
            print(p, sp, ep, rv)
            mv, mc = None, float("Inf")
            for i in range(len(rv) - 1):
                # rvi to sp
                c1 = nx.shortest_path_length(G, rv[i], sp, weight='weight')
                c2 = nx.shortest_path_length(G, ep, rv[i + 1], weight='weight')
                rr1 = nx.shortest_path(G, rv[i], sp, weight='weight')
                rr2 = nx.shortest_path(G, ep, rv[i + 1], weight='weight')
                print(rv[i], rv[i + 1], c1 + c2, rr1, rr2)
                if c1 + c2 < mc:
                    mc = c1 + c2
                    mv = i
            if mc < mmc:
                mmc = mc
                mmv = v
                mmiv = mv
        # print(mmv, mmiv, mmc, stp)
        # update r[mmiv]
        rmmv = routeV[mmv]
        intI = rmmv[mmiv]
        pIS = nx.shortest_path(G, intI, sp, weight='weight')
        intJ = rmmv[mmiv+1]
        pJS = nx.shortest_path(G, ep, intJ, weight='weight')
        newRmmv = rmmv[:(mmiv+1)] + pIS[1:-1] + stp + pJS[1:-1] + rmmv[(mmiv+1):]
        print(rmmv)
        print(intI, pIS)
        print(intJ, pJS)
        print(rmmv)
        print(newRmmv)
        routeV[mmv] = newRmmv
        print()
    for v in routeV:
        print(v, routeV[v])
    return routeV

# greedy passenger insertion" fix insertion using st path (to the same vertex)
def greedy_quadratic(G, lV, lPass, debug=False):
    routeV, routeP = {}, {}
    nv, npass = lV.shape[0], lPass.shape[0]

    for v in range(nv):
        routeV[v] = nx.shortest_path(G, lV[v, 0], lV[v, 1], weight='weight')

    for p in range(npass):
        sp, ep = lPass[p, 0], lPass[p, 1]
        stp = nx.shortest_path(G, sp, ep, weight='weight')
        mmiv1, mmiv2, mmv, mmc = None, None, None, float("Inf")
        for v in range(nv):
            rv = routeV[v]
            mv1, mc1 = None, float("Inf")
            mv2, mc2 = None, float("Inf")
            for i in range(len(rv) - 1):
                c1 = nx.shortest_path_length(G, rv[i], sp, weight='weight')
                if c1 > mc1:
                    continue
                else:
                    mc1 = c1
                    mv1 = i
                for j in range(i + 1, len(rv)):
                    c2 = nx.shortest_path_length(G, ep, rv[j], weight='weight')
                    if c2 < mc2:
                        mc2 = c2
                        mv2 = j
            if mc1 + mc2 < mmc:
                mmv, mmiv1, mmiv2, mmc = v, mv1, mv2, mc1 + mc2
        
        # update r[mmiv]
        if debug:
            print(mmv, mmiv1, mmiv2, mmc)
        rmmv = routeV[mmv]
        intI = rmmv[mmiv1]
        pIS = nx.shortest_path(G, intI, sp, weight='weight')
        intJ = rmmv[mmiv2]
        pJS = nx.shortest_path(G, ep, intJ, weight='weight')
        newRmmv = rmmv[:mmiv1] + pIS + stp[:-1] + pJS[:-1] + rmmv[mmiv2:]
        routeV[mmv] = newRmmv
    return routeV


@click.command()
@click.option("--fname", default="random_rn.pickle", help="RN filename")
@click.option("--npass", default=3, help="# of random passengers")
@click.option("--nv", default=3, help="# of random vehicles")
@click.option("--th", default=400.0, help="request distance threshold")
@click.option("--fshow", is_flag=True, default=False, help="show routing")
def main(fname, npass, nv, th, fshow):
    np.random.seed(1)
    G, pos = pickle.load(open(fname, "rb"))
    
    lPass, used = unique_generator(G, npass, th)
    lV, _ = unique_generator(G, nv, th,used)

    # compute shortest paths
    stP, stV = {}, {}
    for p in range(npass):
        sp, ep = lPass[p, :]
        stp = nx.shortest_path(G, sp, ep, weight='weight')
        stP[p] = stp
    for v in range(nv):
        sv, ev = lV[v, :]
        stv = nx.shortest_path(G, sv, ev, weight='weight')
        stV[v] = stv

    # greedy(V, P)
    # routeV = greedy(G, lV, lPass)
    routeV = greedy_quadratic(G, lV, lPass)

    # debug drawing
    if fshow:
        W = 7
        S = 50
        labels = {n: '[{}]'.format(n) for n in G.nodes}
        fig = plt.figure(figsize=(2 * W, W))
        ax = fig.add_subplot(1, 2, 1)
        nx.draw(G, pos=pos, node_color='k', node_size=10, ax=ax)
        nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax, font_color='red')

        ax = fig.add_subplot(1, 2, 2)
        nx.draw(G, pos=pos, node_color='k', node_size=10, ax=ax)

        # pass
        xylPassO = pos[lPass[:, 0]]
        xylPassD = pos[lPass[:, 1]]
        ax.scatter(xylPassO[:, 0], xylPassO[:, 1], color=['r', 'g', 'b'][:npass], s=S)
        ax.scatter(xylPassD[:, 0], xylPassD[:, 1], color=['r', 'g', 'b'][:npass], marker='s', s=S)

        # vehicles
        xylVO = pos[lV[:, 0]]
        xylVD = pos[lV[:, 1]]
        ax.scatter(xylVO[:, 0], xylVO[:, 1], color=['c', 'm', 'y'][:nv], s=S)
        ax.scatter(xylVD[:, 0], xylVD[:, 1], color=['c', 'm', 'y'][:nv], marker='s', s=S)

        for idv, v in enumerate(range(nv)):
            plot_stpath(ax, pos, routeV[v], color=['c', 'm', 'y'][idv])

        for idv, v in enumerate(range(nv)):
            plot_stpath(ax, pos, stV[v], color=['c', 'm', 'y'][idv], alpha=0.2, lw=5)

        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
