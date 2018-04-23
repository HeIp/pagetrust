# coding=utf-8
import networkx as nx
import numpy as np
from copy import deepcopy
# Importing to log the time ---------------------------
import time
# importing pycuda-------------------------------------
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# ------------------------------------------------------
#
# The PageTrust algorithm:
# How to rank web pages when negative links are allowed?
# http://perso.uclouvain.be/paul.vandooren/publications/deKerchoveV08b.pdf
#
#  *** warning ***
# this code is NOT tested yet.

# debug?
debug_flag = 1


def visualize(name, array):
    if debug_flag == 0:
        return
    print "===== " + name + " ====="
    print array


def initialize_P(g, negative):
    N = len(g)
    P = np.zeros([N, N])
    for item in negative:
        P[item[0]][item[1]] = 1
    tildeP = deepcopy(P)
    visualize('initial P', P)
    return P, tildeP


def build_transition_matrix(alpha, x, g, m):
    N = len(g)
    T = np.zeros([N, N])
    for i in range(N):
        s = 0
        for k in range(N):
            if (k, i) not in g.edges():
                continue
            s += x[k] / g.out_degree(k, weight='weight')
        denominator = alpha * s + (1 - alpha) * 1 / float(N)
        for j in range(N):
            if i == j:
                continue
            if (j, i) not in g.edges():
                continue
            numerator = alpha * g[j][i]['weight'] * x[j] / g.out_degree(j, weight='weight') + m * (1 - alpha) * (
                    1 / float(N)) * x[j]
            T[i][j] = numerator / denominator
    visualize('T', T)
    return T


def is_converged(x1, x2):
    m = 0
    for i in range(len(x1)):
        if (x1[i] - x2[i]) ** 2 > m:
            m = (x1[i] - x2[i]) ** 2
    return m


def calc(g, negative, alpha, m, beta=1):
    epsilon = 0.000000001
    print "start calc pagetrust, epsilon =", epsilon
    N = len(g)
    x = np.ones(N)
    x = x * 1 / N
    visualize("x", x)
    P, tildeP = initialize_P(g, negative)
    # t = 0 is not used ever in the code
    G = nx.google_matrix(g)
    # Underlined code is never used...?
    # pagerank = nx.pagerank(g, alpha=alpha)
    visualize("Google matrix", G)
    t = 0
    while True:
        t += 1
        # build the transition matrix T
        print "***"
        print "*** iteration start, time = ", t
        print "***"
        T = build_transition_matrix(alpha, x, g, m)
        tildeP = np.dot(T, P)
        visualize("P", P)
        visualize("tildeP", tildeP)
        x2 = np.zeros(N)
        # *******************************TIME*****************************************************MARTYN
        start_time = time.time()
        for i in range(N):
            p = 0
            for k in range(N):
                p += G[k, i] * x[k]
            x2[i] = (1 - tildeP[i][i]) ** beta * p
            for j in range(N):
                if (i, j) in negative:
                    P[i, j] = 1
                elif i == j:
                    P[i, j] = 0
                else:
                    P[i, j] = tildeP[i, j]
        # *******************************TIME**************************MARTYN***************************
        print("THIS IS HOW LONG IT TOOK--- %s seconds ---" % (time.time() - start_time))
        # using cuda MARTYN-------------------------------------------------------------------
        print("AND WITH CUDA IT TOOK--- %s seconds ---")
        G = G.astype(np.float32)
        x = x.astype(np.float32)
        P = P.astype(np.float32)
        tildeP = tildeP.astype(np.float32)
        G_gpu = cuda.mem_alloc(G.nbytes)
        x_gpu = cuda.mem_alloc(x.nbytes)
        P_gpu = cuda.mem_alloc(P.nbytes)
        tildeP_gpu = cuda.mem_alloc(tildeP.nbytes)
        cuda.memcpy_htod(G_gpu, G)
        cuda.memcpy_htod(x_gpu, x)
        cuda.memcpy_htod(P_gpu, P)
        cuda.memcpy_htod(tildeP_gpu, tildeP)
        mod = SourceModule("""
        __global__ void cudathreading(float* G_gpu,float* x_gpu,float* P_gpu,float* tildeP_gpu,float* negative,long N) {
            long i = blockIdx.x*blockDim.x + threadIdx.x;
            if (element < N) {
            p = 0
            for( int k = 0; k <= N - 1; k++)
            {
                p += G_gpu[k, i] * x_gpu[k];
                x2[i] = pow( (1 - tildeP_gpu[i][i]) ,beta ) * p;

                for(int j = 0; j <= N - 1; j++)
                {
                    if(i, j) in negative
                    {
                        P_gpu[i, j] = 1;
                    }
                    else if(i == j)
                    {
                        P_gpu[i, j] = 0;
                    }
                    else
                    {
                        P_gpu[i, j] = tildeP_gpu[i, j];
                    }
                }
            }
        }

        void gpu(float* a, long N) {
        int numThreads = 1024; // This can vary, up to 1024
        long numCores = N / 1024 + 1;

        float* gpuA;
        cudaMalloc(&gpuA, N*sizeof(float)); // Allocate enough memory on the GPU
        cudaMemcpy(gpuA, a, N*sizeof(float), cudaMemcpyHostToDevice); // Copy array from CPU to GPU
        cudathreading<<<numCores, numThreads>>>(gpuA, N);  // Call GPU Sqrt
        cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy array from GPU to CPU
        cudaFree(&gpuA); // Free the memory on the GPU
        }""")
        # ---------------------------------------------------------------------------------------------------------------------------------
        # normalization
        tmpl = 0
        for l in range(N):
            tmpl += x2[l]
        for o in range(N):
            x2[o] = x2[o] / tmpl
        visualize("x2", x2)
        e = is_converged(x, x2)
        print "e:", e
        if e < epsilon:
            # visualize('pagerank',pagerank)
            break
        else:
            # x <- x(t+1)
            for p in range(N):
                x[p] = x2[p]
    print x2
    return x2


def test():
    g = nx.DiGraph()
    g.add_weighted_edges_from([(1, 0, 1)])
    g.add_weighted_edges_from([(0, 2, 1)])
    g.add_weighted_edges_from([(2, 0, 1)])
    g.add_weighted_edges_from([(1, 2, 1)])
    g.add_weighted_edges_from([(2, 1, 1)])
    g.add_weighted_edges_from([(2, 3, 1)])
    g.add_weighted_edges_from([(3, 4, 1)])
    g.add_weighted_edges_from([(4, 3, 1)])
    g.add_weighted_edges_from([(4, 1, 1)])
    calc(g, [(0, 3)], 0.85, 0, 1)


if __name__ == "__main__":
    test()
