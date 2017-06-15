
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

dataset=zeros([1000,2])
dataset[:200,]=array([s1])
dataset[200:400,]=array([s2])
dataset[400:600,]=array([s3])
dataset[600:800,]=array([s4])
dataset[800:,]=array([s5])

np.savetxt('dataset.txt', dataset, fmt='%.2e')