import numpy as np
import matplotlib.pyplot as plt


def euclidean(vector1,vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2,2)))

def randomcent(k):
    n = dataset.shape[1]
    cent = np.mat(np.zeros((k, n)))
    for j in range(n):
        lolimit = min(dataset[:, j])
        datarange = float(max(dataset[:, j]) -lolimit)
        cent[:,j] = np.array(lolimit + datarange * np.random.rand(k, 1))
    return cent

def k_means(dataset,k,cent=randomcent):
    m=dataset.shape[0]
    centreplot=cent(k)
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
    return  np.exp(-alpha*(np.linalg.norm(pt1-pt2)**2))

def calc_affinity(Mat):
    Msize = Mat.shape[0]
    affMat = np.zeros([Msize,Msize])
    for i in range(Msize):
        for j in range(Msize):
            affMat[i,j] = affPoint(Mat[i,:],Mat[j,:],0.5)

    return  affMat
def calcLapM(affMat,mode=1):
    if mode== 0:
        LaplMat  = np.diag(np.sum(affMat,axis=1)) - affMat  #row
    elif mode ==1:
        D = np.diag(np.sqrt(np.sum(affMat, axis=1)))
        LaplMat =  np.dot(np.dot(np.linalg.inv(D), affMat), D)
    return  LaplMat

def calc_eigen(Mat):
    eigVal,eigeVec = np.linalg.eig(Mat)
    return  eigeVec

def project(Mat,vec):
    Msize = Mat.shape
    ProjMat = np.dot(Mat,vec)
    return  ProjMat

def NormFeature(FMat):
    rows = FMat.shape[0]
    for k in range(rows):
        FMat[k,:] = FMat[k,:] / np.sqrt(np.sum(FMat[k:]**2))
    return FMat
def spectrum_clustring(Mat,mode):
    affMat = calc_affinity(Mat)
    LaplaseMat = calcLapM(affMat,mode)
    eigVec = calc_eigen(LaplaseMat)
    #prjval = project(Mat,eigVec)
    return  eigVec[:,:10]

if __name__=='__main__':
    sampleNo = 200
    mu1 = np.array([[1, -1]])
    sigma1 = np.array([[1, 0], [0, 1]])
    R1 = np.linalg.cholesky(sigma1)
    s1 = np.dot(np.random.randn(sampleNo, 2), R1) + mu1

    mu2 = np.array([[5.5, -4.5]])
    sigma2 = np.array([[1, 0], [0, 1]])
    R2 = np.linalg.cholesky(sigma2)
    s2 = np.dot(np.random.randn(sampleNo, 2), R2) + mu2

    mu3 = np.array([[1, 4]])
    sigma3 = np.array([[1, 0], [0, 1]])
    R3 = np.linalg.cholesky(sigma3)
    s3 = np.dot(np.random.randn(sampleNo, 2), R3) + mu3

    mu4 = np.array([[6, 4.5]])
    sigma4 = np.array([[1, 0], [0, 1]])
    R4 = np.linalg.cholesky(sigma4)
    s4 = np.dot(np.random.randn(sampleNo, 2), R4) + mu4

    mu5 = np.array([[9, 0.0]])
    sigma5 = np.array([[1, 0], [0, 1]])
    R5 = np.linalg.cholesky(sigma5)
    s5 = np.dot(np.random.randn(sampleNo, 2), R5) + mu5

    plt.scatter(s1[:, 1], s1[:, 0], marker='D', color='c')
    plt.scatter(s2[:, 1], s2[:, 0], marker='*', color='m')
    plt.scatter(s3[:, 1], s3[:, 0], marker='o', color='b')
    plt.scatter(s4[:, 1], s4[:, 0], marker='+', color='r')
    plt.scatter(s5[:, 1], s5[:, 0], marker='x', color='g')

    dataset = np.zeros([1000, 2])
    dataset[:200, ] = np.array([s1])
    dataset[200:400, ] = np.array([s2])
    dataset[400:600, ] = np.array([s3])
    dataset[600:800, ] = np.array([s4])
    dataset[800:, ] = np.array([s5])
    k = np.int16(5)
    eigvec = spectrum_clustring(dataset[::5,:],mode=0)
    #np.savetxt('eigenvec.txt',eigvec,fmt='%.2e')

    plt.figure()

    Feature = NormFeature(eigvec[:,:2].copy())
    plt.scatter(Feature[:, 1], Feature[:, 0])
    np.savetxt('posrnorn.txt',Feature,fmt='%.2e')
    label = k_means(Feature,k)
    center = randomcent(k)

    # for elfa in range(k):
    #     ptsInClust = dataset[np.nonzero(label[:, 0].getA() == elfa)[0]]
    #     center[elfa, :] = np.mean(ptsInClust, axis=0)
    # plt.scatter(center[:, 1], center[:, 0], s=300, marker='H', color='k')

    #print center
    plt.figure()
    plt.subplot(230)
    plt.plot(eigvec[:,0])
    plt.subplot(231)
    plt.plot(eigvec[:,1])
    plt.subplot(232)
    plt.plot(eigvec[:,2])
    plt.subplot(233)
    plt.plot(eigvec[:,3])
    plt.subplot(234)
    plt.plot(eigvec[:,4])

    plt.show()
