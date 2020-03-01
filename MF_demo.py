# !/user/bin/env python
# encoding:utf-8
from math import pow
import numpy

def matrix_factorization(R,P,Q,K,steps=5000,alpha=0.0002,beta=0.02):
    Q = Q.T
    result =[]
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] = numpy . dot(P[i,:],Q[:,j])     #矩阵内积
                    for k in range(K):
                        P[i][k]=P[i][k] + alpha*(2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j]=Q[k][j] + alpha*(2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]),2)
        result.append(e)
        if e < 0.001:
            break
    return P,Q.T,result



R = [
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4]
    ]

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

# 随机生成一个N行K列和M行K列的矩阵
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

print("原始的评分矩阵 R 为：\n" ,R)

nP,nQ,result = matrix_factorization(R,P,Q,K)

print("原始的评分矩阵 R 为：\n" ,R)


R_MatrixFac = numpy.dot(nP,nQ.T)
print("经过矩阵分解后，对0处的评分进行了填充，得到新的评分矩阵R_MatrixFac :\n",R_MatrixFac)

