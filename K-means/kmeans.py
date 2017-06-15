
from numpy import *
from matplotlib.pyplot import *

sampleNo = 200
mu1 = array([[1, -1]])
sigma1 =array([[1,0],[0,1]])
R1 = linalg.cholesky(sigma1)
s1 = dot(random.randn(sampleNo, 2), R1) + mu1

mu2 = array([[5.5, -4.5]])
sigma2 = array([[1,0],[0,1]])
R2 = linalg.cholesky(sigma2)
s2 = dot(random.randn(sampleNo, 2), R2) + mu2

mu3 = array([[1, 4]])
sigma3 = array([[1,0],[0,1]])
R3 = linalg.cholesky(sigma3)
s3 = dot(random.randn(sampleNo, 2), R3) + mu3

mu4 = array([[6, 4.5]])
sigma4 = array([[1,0],[0,1]])
R4 = linalg.cholesky(sigma4)
s4 = dot(random.randn(sampleNo, 2), R4) + mu4

mu5 = array([[9, 0.0]])
sigma5 = array([[1,0],[0,1]])
R5 = linalg.cholesky(sigma5)
s5 = dot(random.randn(sampleNo, 2), R5) + mu5

scatter(s1[:,1],s1[:,0],marker = 'D', color ='c')
scatter(s2[:,1],s2[:,0],marker = '*', color ='m')
scatter(s3[:,1],s3[:,0],marker = 'o',color ='b')
scatter(s4[:,1],s4[:,0],marker = '+',color ='r')
scatter(s5[:,1],s5[:,0],marker = 'x',color ='g')

dataset=zeros([1000,2])
dataset[:200,]=array([s1])
dataset[200:400,]=array([s2])
dataset[400:600,]=array([s3])
dataset[600:800,]=array([s4])
dataset[800:,]=array([s5])


def euclidean(vector1,vector2):
    return sqrt(sum(power(vector1-vector2,2)))

def randomcent(k):
    n = dataset.shape[1]
    cent = mat(zeros((k, n)))
    for j in range(n):
        lolimit = min(dataset[:, j])
        datarange = float(max(dataset[:, j]) -lolimit)
        cent[:,j] = array(lolimit + datarange * random.rand(k, 1))
    return cent

def k_means(dataset,k,cent=randomcent):
    m=dataset.shape[0]
    centreplot=cent(k)
    evaluate=mat(zeros([m,2]))
    clusterchange = True
    while clusterchange:
        clusterchange = False
        for i in range(m):
            mineuclid = inf
            mincentidx=-1
            for j in range(k):
                eucliJ=euclidean(dataset[i,], centreplot[j, :])
                if eucliJ<mineuclid:
                    mincentidx=j
                    mineuclid=eucliJ
            if evaluate[i, 0] != mincentidx: clusterchange = True
            evaluate[i, :] = [mincentidx, mineuclid **2]
        for elfa in range(k):
            ptsInClust = dataset[nonzero(evaluate[:, 0].getA() == elfa)[0]]
            centreplot[elfa, :] = mean(ptsInClust, axis=0)
    scatter(centreplot[:, 1], centreplot[:, 0], marker='H', color='k')
    return centreplot,evaluate

k_means(dataset,5)
show()


