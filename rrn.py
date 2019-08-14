# -*- coding: utf-8 -*-
#
# build random road network
#
# 1. randomly sample location points
# 2. compute pair-wise shortest paths
# 3. keep pairs if its length < 1.5 x (ST)

import pickle
import numpy as np
import networkx as nx
import numpy.random as npr
import numpy.linalg as npl
from collections import defaultdict
import matplotlib.pyplot as plt

# np.random.seed(0)

def generate(D, N, coeff):
    xy = np.random.rand(N, 2) * D

    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca()
    ax.scatter(x=xy[:, 0], y=xy[:, 1])
    plt.tight_layout()
    plt.show()
    plt.close()

    mindist = float("Inf")
    distlist = []
    # distdict = defaultdict(float)
    for i in range(N):
        for j in range(i + 1, N):
            dij = npl.norm(xy[i, :] - xy[j, :])
            # distdict[(i, j)] = dij
            distlist.append((i, j, dij))
            mindist = min(mindist, dij)
    distlist = sorted(distlist, key=lambda x: x[2])
    
    # min ST and 1.5x procedure
    # G = nx.DiGraph()
    G = nx.Graph()
    reminder = []
    ijdij = distlist[0]
    reminder.append((ijdij[0], ijdij[1]))
    distlist.remove(ijdij)

    G.add_edge(ijdij[0], ijdij[1], weight=ijdij[2])
    for ijdij in distlist:
        # check already connected
        u, v, duv = ijdij
        try:
            pathuv = nx.shortest_path(G, u, v, weight='weight')
        except nx.exception.NodeNotFound as e:
            pathuv = []
            pass
        except nx.exception.NetworkXNoPath as e:
            pathuv = []
            pass
        finally:
            if not pathuv:
                G.add_edge(u, v, weight=duv)
            else:
                dpathuv = nx.shortest_path_length(G, u, v, weight='weight')
                if dpathuv > duv * coeff:
                    G.add_edge(u, v, weight=duv)

    # output
    print(G.number_of_nodes(), G.number_of_edges())
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca()
    nx.draw(G, pos=xy, ax=ax, node_size=10)
    plt.tight_layout()
    plt.show()
    plt.close()

    return G, xy

if __name__ == '__main__':
    D = 800
    N = 100
    coeff = 1.5
    G, pos = generate(D, N, coeff)
    with open("random_rn.pickle", "wb") as f:
        pickle.dump((G, pos), f)