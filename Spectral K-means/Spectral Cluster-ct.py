import numpy as np
import matplotlib.pyplot as plt

def euclidean(vector1,vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2,2)))

def randomcent(dataset,k):
    n = dataset.shape[1]
    cent = np.mat(np.zeros((k, n)))
    for j in range(n):
        lolimit = min(dataset[:, j])
        datarange = float(max(dataset[:, j]) -lolimit)
        cent[:,j] = np.array(lolimit + datarange * np.random.rand(k, 1))
    return cent

def k_means(dataset,k,cent=randomcent):
    m=dataset.shape[0]
    centreplot=cent(dataset,k)
    evaluate=np.mat(np.zeros([m,2]))
    clusterchange = True
    while clusterchange:
        clusterchange = False
        for i in range(m):
            mineuclid = np.inf
            mincentidx=-1
            for j in range(k):
                eucliJ=euclidean(dataset[i,], centreplot[j, :])
                if eucliJ<mineuclid:
                    mincentidx=j
                    mineuclid=eucliJ
            if evaluate[i, 0] != mincentidx:
                clusterchange = True
            evaluate[i, :] = [mincentidx, mineuclid **2]
        for elfa in range(k):
            ptsInClust = dataset[np.nonzero(evaluate[:, 0].getA() == elfa)[0]]
            centreplot[elfa, :] = np.mean(ptsInClust, axis=0)
    return  evaluate

def affPoint(pt1,pt2,alpha):
    return  np.exp(-alpha*(np.linalg.norm(pt1-pt2)**2))     #sigma=sqrt(1/2alpha)

def calc_affinity(Mat):
    Msize = Mat.shape[0]
    affMat = np.zeros([Msize, Msize])
    for i in range(Msize):
        ptseucli = np.zeros([Msize, 2])
        for j in range(Msize):
            ptseucli[j,0] =euclidean(Mat[i,:],Mat[j,:])
            ptseucli[j,1] =j
        order = np.argsort(ptseucli[:, 0])
        ptseucli=ptseucli[order,:]
        for j in range(9):                                          #k-nearst,k=8
            idxj = ptseucli[j,1]
            affMat[i,idxj] = affPoint(Mat[i, :], Mat[idxj, :], 0.5)  # choose sigma=1
        affMat[i,i]=0
    return  affMat

def calc_LapM(affMat):                                        #laplace
    D=np.diag(np.sum(affMat,axis=1))
    sqrtD=np.diag(np.sqrt(np.sum(affMat, axis=1)))            #axis=1  row
    Msize=D.shape[0]
    invD=np.zeros([Msize,Msize])
    for i in range(Msize):
        if sqrtD[i,i]!=0:
            invD[i,i]=1/sqrtD[i,i]
        else:
            invD[i,i]=sqrtD[i,i]
    LaplMat=D-affMat
    LaplMat =  np.dot(invD, np.dot(LaplMat, sqrtD))
    return  LaplMat

def calc_eigenvec(Mat):
    eigVal,eigeVec = np.linalg.eig(Mat)
    order = np.argsort(np.abs(eigVal))
    eigeVec=eigeVec[:,order].real
    return eigeVec[:,0:2]

def Normaliaze(FMat):
    rows = FMat.shape[0]
    for k in range(rows):
        FMat[k,:] = FMat[k,:] / np.linalg.norm(FMat[k,:])
    return FMat

def spectrum_clustring(Mat):
    n=Mat.shape[0]
    affMat = calc_affinity(Mat)
    LaplaseMat = calc_LapM(affMat)
    Uvec = calc_eigenvec(LaplaseMat)
    spcidx=k_means(Uvec,2)
    ptsInSpClust1 = Mat[np.nonzero(spcidx[:, 0].getA() == 0)[0]]
    plt.scatter(ptsInSpClust1[:,0],ptsInSpClust1[:,1], marker='x', color='g')
    ptsInSpClust2 = Mat[np.nonzero(spcidx[:, 0].getA() == 1)[0]]
    plt.scatter(ptsInSpClust2[:,0],ptsInSpClust2[:,1], marker='h', color='r')
    n1M = np.nonzero(spcidx[:, 0].getA() == 0)[0]
    n2M = np.nonzero(spcidx[:, 0].getA() == 1)[0]
    n1 = 0
    n1T = 0
    n2 = 0
    n2T = 0
    i1=n1M.shape[0]
    for i in range(i1):
        if n1M[i] > 99:
            n1 += 1
        else:
            n1T += 1
    i2=n2M.shape[0]
    for j in range(i2):
        if n2M[j] < 100:
            n2 += 1
        else:
            n2T += 1
    n1 = max(n1, n1T)
    n2 = max(n2, n2T)
    Accu = (n1 + n2) / float(n)
    print  Accu
dataset = np.loadtxt('dataset.txt')
spectrum_clustring(dataset)
plt.show()
