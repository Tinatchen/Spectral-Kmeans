from numpy import *
from matplotlib.pyplot import *

def euclidean(vector1,vector2):
    return sqrt(sum(power(vector1-vector2,2)))

def randomcent(k):
    n = dataset.shape[1]
    cent = mat(zeros([k, n]))
    for j in range(n):
        lolimit = min(dataset[:, j])
        datarange = float(max(dataset[:, j]) -lolimit)
        cent[:,j] = array(lolimit + datarange * random.rand(k, 1))
    return cent

def k_means(dataset,k,cent=randomcent):
    m=dataset.shape[0]
    centreplot=cent(k)  #first
    evaluate=mat(zeros([m,2]))
    ptsamount=array(zeros([k,1]))
    for i in range(m):
        mineuclid = inf
        mincentidx = -1
        for j in range(k):
            eucliJ = euclidean(dataset[i,], centreplot[j, :])
            if eucliJ < mineuclid:
                mincentidx = j
                mineuclid = eucliJ
        evaluate[i, :] = [mincentidx, mineuclid ** 2]
    for elfa in range(k):
        ptsInClust = dataset[nonzero(evaluate[:, 0].getA() == elfa)[0]]
        centreplot[elfa, :] = mean(ptsInClust, axis=0)              #classify x
        ptsamount[elfa,]=ptsInClust.shape[0]               #acount nj
    clusterchange=True
    while clusterchange:
        clusterchange = False
        for i in range(m):
            minpj=inf
            minpjidx=-1
            ptsidx=evaluate[i,0]
            for j in range(k):
                nj=ptsamount[j,:]
                euc = euclidean(dataset[i,], centreplot[j,])
                if ptsidx==j:
                    pj=nj*euc/(nj-1) - 1e-1                          #invoid oscillation
                else:
                    pj=nj*euc/(nj+1)
                if pj<minpj:
                    minpj=pj
                    minpjidx=j
            if evaluate[i,0]!=minpjidx:
                clusterchange=True
            evaluate[i, :] = [minpjidx, euc ** 2]
        for elfa in range(k):
            ptsInClust = dataset[nonzero(evaluate[:,0].getA() == elfa)[0]]
            centreplot[elfa, :] = mean(ptsInClust, axis=0)
            ptsamount[elfa,] = ptsInClust.shape[0]
        clf()
        dr1 = dataset[nonzero(evaluate[:, 0].getA() == 0)[0]]
        dr2 = dataset[nonzero(evaluate[:, 0].getA() == 1)[0]]
        dr3 = dataset[nonzero(evaluate[:, 0].getA() == 2)[0]]
        dr4 = dataset[nonzero(evaluate[:, 0].getA() == 3)[0]]
        dr5 = dataset[nonzero(evaluate[:, 0].getA() == 4)[0]]

        scatter(dr1[:, 1], dr1[:, 0], marker='D', color='c')
        scatter(dr2[:, 1], dr2[:, 0], marker='*', color='m')
        scatter(dr3[:, 1], dr3[:, 0], marker='o', color='b')
        scatter(dr4[:, 1], dr4[:, 0], marker='+', color='r')
        scatter(dr5[:, 1], dr5[:, 0], marker='x', color='g')
        draw()
        pause(0.001)
    print ptsamount.T
    centreplot = np.array(centreplot)
    print centreplot
    scatter(centreplot[:, 1], centreplot[:, 0],s=300, marker='H', color='k')
    # evaluate[i, 1] = euclidean(dataset[i,]-centreplot[ptsidx,])

dataset = loadtxt('dataset.txt')
k_means(dataset,5)
show()
