import networkx as nx
import numpy as np
import time
from copy import deepcopy


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#
#The PageTrust algorithm:
#How to rank web pages when negative links are allowed?
#http://perso.uclouvain.be/paul.vandooren/publications/deKerchoveV08b.pdf
#
#  *** warning ***
# this code is NOT tested yet.

#debug?
debug_flag = 1

def visualize(name,array):
	if debug_flag == 0:
		return
	print "===== " + name + " ====="
	print array

def initialize_P(g):
	N = len(g)
	P = np.zeros([N,N])
	P[0][3] = 1
	tildeP = deepcopy(P)
	visualize('initial P',P)
	return P,tildeP

def build_transition_matrix(alpha,x,g,G,M):
	N = len(g)
	T = np.zeros([N,N])
	for i in range(N):
		s = 0
		for k in range(N):
			if (k,i) not in g.edges():
				continue
			s += x[k] / g.out_degree(k,weight='weight')
		denominator = alpha * s + (1 - alpha) * 1/float(N)
		for j in range(N):
			if i == j:
				continue
			if (j,i) not in g.edges():
				continue
			numerator = alpha * g[j][i]['weight'] * x[j] / g.out_degree(j,weight='weight') + M * (1 - alpha)*(1/float(N))*x[j]
			T[i][j] = numerator/denominator
	visualize('T',T)
	return T

def is_converged(x1,x2):
	m = 0
	for i in range(len(x1)):
		if (x1[i] - x2[i])**2 > m:
			m = (x1[i] - x2[i])**2
	return m

def calc(g,alpha,M,beta=1):
        N = np.int32(len(g))
        negative = np.zeros([N*N])
        negative[0*N+3] = 1
	epsilon = 0.000000001
	print "start calc pagetrust, epsilon =",epsilon
	x = np.ones(N)
	x = x * 1/N
	visualize("x",x)
	P,tildeP = initialize_P(g)
	t = 0
	G = nx.google_matrix(g)
	pagerank = nx.pagerank(g,alpha=alpha)
	visualize("Google matrix",G)
	t = 0
	while True:
		t += 1
		#build the transition matrix T
		print "***"
		print "*** iteration start, time = ",t
		print "***"
		T = build_transition_matrix(alpha,x,g,G,M)
		tildeP = np.dot(T,P)
		visualize("P",P)
		visualize("tildeP",tildeP)
		x2 = np.zeros(N)
		print("x2 shape: ", x2.shape)

		# Step1: Flatten P & G & tildeP, print shape
		fg = G.reshape((1, -1))
		fg = np.asarray(fg)[0]

		fp = P.flatten()
		ftp = tildeP.flatten()

		print("flat P shape: ", fp.shape)
		print("flat tildeP shape: ", ftp.shape)
		print("flat G shape: ", fg.shape)


		overallTimeElapsed = 0


         #------------------------------PYCUDA STARTS BEING USED-----------------------
                negative = negative.astype(np.float32)
	        fg = fg.astype(np.float32)
                x = x.astype(np.float32)
                ftp = ftp.astype(np.float32)
                G_gpu = cuda.mem_alloc(fg.nbytes)
                fp = fp.astype(np.float32)
                x2 = x2.astype(np.float32)
                negative_gpu = cuda.mem_alloc(negative.nbytes)
                x2_gpu = cuda.mem_alloc(x2.nbytes)
                x_gpu = cuda.mem_alloc(x.nbytes)
                fp_gpu = cuda.mem_alloc(fp.nbytes)
                ftp_gpu = cuda.mem_alloc(ftp.nbytes)
                cuda.memcpy_htod(G_gpu, fg)
                cuda.memcpy_htod(x_gpu, x)
                cuda.memcpy_htod(fp_gpu, fp)
                cuda.memcpy_htod(ftp_gpu, ftp)
                cuda.memcpy_htod(x2_gpu, x2)
                start_time = time.time()
                mod = SourceModule("""
         __global__ void cudathreading(float* G_gpu, float* x_gpu, float* fp_gpu, float* ftp_gpu, float* negative_gpu, float* x2_gpu, int N) {
            int beta= 1;
            long i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < N) {
            int p = 0;
            for( int k = 0; k <= N - 1; k++)
            {
            p += G_gpu[k*N+i] * x_gpu[k];
            x2_gpu[i] = pow( (1 - ftp_gpu[i*N+i]) ,beta ) * p;
    
            for(int j = 0; j <= N - 1; j++)
            {
                if(negative_gpu[i,j] < 0)
                {
                    fp_gpu[i*N+j] = 1;
                }
                else if(i == j)
                {
                    fp_gpu[i*N+j] = 0;
                }
                else
                {
                    fp_gpu[i*N+j] = ftp_gpu[i, j];
                }
            }
        }
    }
 }   
    
    """)
                func = mod.get_function("cudathreading")
                func(G_gpu, x_gpu, fp_gpu, ftp_gpu, negative_gpu, x2_gpu, N, block=(5,1,1))
                # *******************************TIME**************************MARTYN***************************
                print("THIS IS HOW LONG IT TOOK--- %s seconds ---" % (time.time() - start_time))
        
         # -------------------------PYCUDA STOPS BEING USED-------------------------------------------------
	 #normalization
                tmpl = 0
                for l in range(N):
                    tmpl += x2[l]
                for o in range(N):
                    x2[o] = x2[o] / tmpl
                visualize("x2",x2)
                e = is_converged(x,x2)
                print "e:",e
                if e < epsilon:
                    #visualize('pagerank',pagerank)
                    break
                else:
                    #x <- x(t+1)
                    for p in range(N):
                        x[p] = x2[p]
	
                print x2
                return x2
	

def test():
	g = nx.DiGraph()
	g.add_weighted_edges_from([(1,0,1)])
	g.add_weighted_edges_from([(0,2,1)])
	g.add_weighted_edges_from([(2,0,1)])
	g.add_weighted_edges_from([(1,2,1)])
	g.add_weighted_edges_from([(2,1,1)])
	g.add_weighted_edges_from([(2,3,1)])
	g.add_weighted_edges_from([(3,4,1)])
	g.add_weighted_edges_from([(4,3,1)])
	g.add_weighted_edges_from([(4,1,1)])
	calc(g,0.85,0,1)

if __name__=="__main__":
	test()
