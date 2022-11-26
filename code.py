import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

def plot(A, title = ""):
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(7)
    plt.imshow(A, cmap='hot', interpolation='nearest')
    plt.title(title)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()
    print()

def getUniqueEdges(filename, separator):
    E = []
    s = set()
    KV = {}
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            data = line.split(separator)
            u = int(data[0])
            v = int(data[1])
            E.append([u, v])
            s.add(u)
            s.add(v)
    
    n = len(s)
    cnt = 0
    for x in s:
        KV[x] = cnt
        cnt += 1
    
    A = [[0 for _ in range(n)] for _ in range(n)]
    E1 = []
    
    for e in E:
        u = KV[e[0]]
        v = KV[e[1]]
        
        if A[u][v] == 1:
            continue
        A[u][v] = A[v][u] = 1
        E1.append(e)
    
    E1 = np.array(E1)
    print('#Vertices        = ', n)    
    print("#Edges           = ", len(E))
    # print("#Unique Edges    = ", len(E1))
    return E1
        
def import_facebook_data(filename):
    return getUniqueEdges(filename, ' ')

def import_bitcoin_data(filename):
    return getUniqueEdges(filename, ',')

def spectralDecomp_OneIter(E):
    V = set()
    for e in E:
        V.update(e)
    
    V = list(V)
    n = len(V)
    kv = {}
    for i in range(n):
        kv[V[i]] = i
    
    # computing the Degree Matrix D and Adjacency Matrix A
    A = np.zeros((n, n))
    Deg = [0 for _ in range(n)]
    
    for e in E:
        Deg[kv[e[0]]] += 1
        Deg[kv[e[1]]] += 1
        A[kv[e[0]]][kv[e[1]]] = 1
        A[kv[e[1]]][kv[e[0]]] = 1
    
    D = np.diag(Deg)

    # Graph Laplacian L
    L = D - A
    

    # computing inv(D)L for normalized cut
    v, w = np.linalg.eigh(np.matmul(np.linalg.inv(D), L))
    
    partition = np.zeros((n, 2))
    
    # eigen vector corresponding to second minimum eigen value
    eig_vec = w[:, 1]
    
    # partition nodes based on the sign of eig_vec vector
    cid_neg = math.inf
    cid_pos = math.inf
    for i in range(n):
        if eig_vec[i] <= 0:
            cid_neg = min(cid_neg, V[i])
        else:
            cid_pos = min(cid_pos, V[i])
    
    for i in range(n):
        partition[i][0] = V[i]
        partition[i][1] = (cid_neg if eig_vec[i] <= 0 else cid_pos)

    return eig_vec, A, partition

def spectralRecursive(n, E, cluster, cid):
    if n == 0:
        return cid
    
    # partition Edge set E
    _, _, partition = spectralDecomp_OneIter(E)
    n2p = {}
    
    
    for x in partition:
        n2p[x[0]] = x[1]
    
    # computing E1 and E1 for the edge set of the two partitions
    E1 = []
    E2 = []
    V1 = set()
    V2 = set()
    degV1 = 0
    degV2 = 0
    for e in E:
        if n2p[e[0]] == n2p[e[1]]:
            if n2p[e[0]] == partition[0][1]:
                V1.update(e)
                E1.append(e)
                degV1 += 2
            else:
                V2.update(e)
                E2.append(e)
                degV2 += 2
        else:
            degV1 += 1
            degV2 += 1
    
    # computing ratio for normalized cut
    ratio = 1
    if degV1 > 0 and degV2 > 0:
        cut_e = len(E) - len(E1) - len(E2)
        ratio = (cut_e / degV1) + (cut_e / degV2)
    
    # if ratio > 0.6 don't do partitioning and return
    if ratio > 0.6:
        for e in E:
            cluster[e[0]] = cluster[e[1]] = cid
        return cid + 1
        
    # recursive call to partition edge set E1 and E2
    cid = spectralRecursive(len(V1), E1, cluster, cid)
    cid = spectralRecursive(len(V2), E2, cluster, cid)
    
    return cid
    
def spectralDecomposition(E):

    # preprocessing
    cluster = {}
    for e in E:
        cluster[e[0]] = -1
        cluster[e[1]] = -1
    n = len(cluster)


    cluster_count = spectralRecursive(n, E, cluster, 0)
    
    # post processing
    c2n = {}
    n2c = cluster
    for key in n2c:
        if n2c[key] == -1:
            n2c[key] = key
        if c2n.get(n2c[key]) == None:
            c2n[n2c[key]] = key
        else:
            c2n[n2c[key]] = min(key, c2n[n2c[key]])
    
    # print("Total Partitions = ", len(c2n.keys()))
    V = list(cluster.keys())
    n = len(V)
    partition = np.zeros((n, 2))
    for i in range(n):
        partition[i][0] = V[i]
        partition[i][1] = c2n[n2c[V[i]]]
        
    return partition

def createSortedAdjMat(partition, E):
    V = set()
    for e in E:
        V.update(e)
    V = list(V)
    n = len(V)
    
    kv = {}
    
    for i in range(n):
        kv[V[i]] = i
     
    # creating the adjacency Matrix A from edge set E
    A = np.zeros((n, n))
    for e in E:
        A[kv[e[0]]][kv[e[1]]] = 1
        A[kv[e[1]]][kv[e[0]]] = 1
    
    # arranging each node in increasing order of their community Ids
    p_list = []
    for i in range(n):
        p_list.append([partition[i][1], partition[i][0]])
    p_list.sort()
    
    # creting the adjacency matrix B arranged in increasing order of community Ids
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = kv[p_list[i][1]]
            v = kv[p_list[j][1]]
            B[i][j] = A[u][v]
    
    return B

# change in modularity while moving node u from community x to y
def deltaQ(u, x, y, adj, cluster, m):
    nodes_x = set()
    nodes_x.update(cluster[x])
    nodes_y = set()
    nodes_y.update(cluster[y])
    
    sigma_x = 0
    for v in cluster[x]:
        sigma_x += len(adj[v])
    
    sigma_y = 0
    for v in cluster[y]:
        sigma_y += len(adj[v])
    
    deg_u = len(adj[u])
    # edges of u in x
    e_ux = 0
    # edges of u in y
    e_uy = 0
    for v in adj[u]:
        if v in nodes_x:
            e_ux += 1
        elif v in nodes_y:
            e_uy += 1
   
    # dQx is change in modularity of x after u is removed
    # dQy is change in modularity of y after u is added

    dQx = -2 * e_ux / (2 * m)
    dQy = 2 * e_uy / (2 * m)
    
    m4m = 4 * m * m
    dQx = dQx + (sigma_x ** 2 - (sigma_x - deg_u) ** 2) / m4m
    dQy = dQy + (sigma_y ** 2 - (sigma_y + deg_u) ** 2) / m4m
    
    # returning total modularity change
    return dQx + dQy

def louvain_one_iter(E):
    m = len(E)
    # adjacency list
    adj = {}
    for e in E:
        if e[0] not in adj:
            adj[e[0]] = []
        if e[1] not in adj:
            adj[e[1]] = []
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    
    # node to cluster
    n2c = {}
    for v in adj:
        n2c[v] = v
    
    # initial modularity
    Q = 0
    for u in adj:
        Q += len(adj[u]) ** 2
        
    Q = -Q / 4 / m / m
    
    # print("Initial modularity = ", Q)
    
    V = list(adj.keys())

    # Iterating until modularity is saturated
    for k in range(30):
        count = 0

        # for each node u in Vertex set V
        for u in V:
        
            #######
            # cluster to node
            c2n = {}
            for v in n2c:
                if c2n.get(n2c[v]) == None:
                    c2n[n2c[v]] = []
                c2n[n2c[v]].append(v)

            # track the cluster _c where node _u belongs ((u, _u) element of E)
            # such that delta is maximum so far if node u 
            # is moved to cluster _c
            _u = -1
            _c = -1
            delta = -math.inf
            ######
            
            for v in adj[u]:
                c = n2c[v]
                # if in same cluster or no edge of u with target cluster c
                if n2c[u] == c:
                    continue
                    
                delQ = deltaQ(u, n2c[u], c, adj, c2n, m)
                if delQ > delta:
                    delta = delQ
                    _u = u
                    _c = c
            

            if Q + delta <= Q:
                count += 1
                continue
            
            # updates Q when new modularity is more than previous
            Q = Q + delta
            n2c[_u] = _c
            # print(_u, _c, Q)
        # print(f"Iteration = {k}, Modularity = {Q}")
        if count == len(V):
            break
    
    # print(Q)
    
    # for each cluster, the smallest node it mapped to
    c2n = {}
    for key in n2c:
        if c2n.get(n2c[key]) == None:
            c2n[n2c[key]] = key
        else:
            c2n[n2c[key]] = min(key, c2n[n2c[key]])
    
    # print("Total Partitions = ", len(c2n.keys()))
    n = len(V)
    partition = np.zeros((n, 2))
    for i in range(n):
        partition[i][0] = V[i]
        partition[i][1] = c2n[n2c[V[i]]]
        
    return partition
    

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
