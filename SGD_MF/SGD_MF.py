#!usr/bin/python
# -*- coding:UTF-8 -*-
#Created on: 2018/3/12
#author: Xiuze Zhou
#e-mail: zhouxiuze@foxmail.com
 
#-------------------------FUNCTION---------------------------#
from pylab import *
import numpy as np
import random
import math

def SGD(train,test,N,M,gamma,D,lambda_1,Step):
    # train: train data
    # test: test data
    # N:the number of user
    # M:the number of item
    # gamma: the learning rata
    # D: the number of latent factor
    # lambda_1: regularization parameter
    # Step: the max iteration
    p = np.random.random((N, D))
    q = np.random.random((M, D))
    rmse=[]
    loss=[]
    for ste in range(Step):
        los=0.0
        for data in train:
            u=data[0]
            i=data[1]
            r=data[2]

            e=r-np.dot(p[u],q[i].T)            
            p[u]=p[u]+gamma*(e*q[i]-lambda_1*p[u])
            q[i]=q[i]+gamma*(e*p[u]-lambda_1*q[i])

            los=los+e**2+lambda_1*(np.square(p[u]).sum()+np.square(q[i]).sum())
        loss.append(los)
        rms=RMSE(p,q,test)
        rmse.append(rms)
        if ste%10==0:
            print ste/10
    return loss,rmse,p,q

           
def RMSE(p,q,test):
    count=len(test)
    sum_rmse=0.0
    for t in test:
        u=t[0]
        i=t[1]
        r=t[2]
        pr=np.dot(p[u],q[i].T)
        sum_rmse+=np.square(r-pr)
    rmse=np.sqrt(sum_rmse/count)
    return rmse


def Load_data(filedir,ratio):
    user_set={}
    item_set={}
    N=0;#the number of user
    M=0;#the number of item
    u_idx=0
    i_idx=0
    data=[]
    f = open(filedir)
    for line in f.readlines():
        u,i,r,t=line.split()
        if int(u) not in user_set:
            user_set[int(u)]=u_idx
            u_idx+=1
        if int(i) not in item_set:
            item_set[int(i)]=i_idx
            i_idx+=1
        data.append([user_set[int(u)],item_set[int(i)],int(r)])
    f.close()
    N=u_idx;
    M=i_idx;

    np.random.shuffle(data)
    train=data[0:int(len(data)*ratio)]
    test=data[int(len(data)*ratio):]
    return N,M,train,test


def Figure(loss,rmse):
    fig1=plt.figure('LOSS')
    x = range(len(loss))
    plot(x, loss, color='g',linewidth=3)
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    fig2=plt.figure('RMSE')
    x = range(len(rmse))
    plot(x, rmse, color='r',linewidth=3)
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    show()

#----------------------------SELF TEST----------------------------#
 
def main():
    dir_data="./u.data"
    ratio=0.8
    N,M,train,test=Load_data(dir_data,ratio)
        
    gamma=0.005
    D=10
    lambda_1=0.1
    Step=50
    loss,rmse,p,q=SGD(train,test,N,M,gamma,D,lambda_1,Step)
    print rmse[-1];
    Figure(loss,rmse)
    
         
if __name__ == '__main__': 
    main()
