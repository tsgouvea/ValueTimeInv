
# import sys
import os
import pickle

import pandas as pd
import numpy as np
from scipy.special import expit, softmax, lambertw
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
import seaborn as sns

from locallib import makephi, pnorm, truncExp
#TODO: complete this exercise
#TODO: compute E[G3] properly

def EG1(pB,K,beta,tau):
    G1 = pB * (1 - np.exp(-K / beta)) / (K + tau)
    return G1

def kEG1(beta,tau):
    k = np.real(-beta * lambertw(-np.exp(-(beta + tau) / beta), k=-1) - beta - tau)
    if np.isscalar(k):
        k = k if k>0 else 0
    else:
        k[k<0]=0
    return k

def EG2(pB,K,beta,tau,rho):
    G2 = pB * (1 - np.exp(-K / beta)) - rho*(K + tau)
    return G2

def kEG2(beta,rho,pB):
    k = np.real(beta * np.log(pB/(rho*beta)))
    if np.isscalar(k):
        k = k if k>0 else 0
    else:
        k[k<0]=0
    return k

def EG3(pB,K,beta,tau,rho,m=1):
    e=np.exp(-K/beta)
    m=1
    G3 = pB * (1 - e) * (m - rho * (tau * beta)) + (1 - pB * (1 - e)) * (-rho * (tau * K))
    return G3

def kEG3(beta,rho,pB,m=1):
    k = (-beta * rho * lambertw((pB - 1) * np.exp((2 * beta * rho - m) / (beta * rho)) / pB, k=-1) + 2 * beta * rho - m) / rho
    if np.isscalar(k):
        k = k if k>0 else 0
    else:
        k[k<0]=0
    return k

def kLak14(beta,rho,pB):
    k = np.real(beta * np.log(pB/(1-pB)*(1-rho*beta)/rho*beta))
    if np.isscalar(k):
        k = k if k>0 else 0
    else:
        k[k<0]=0
    return k
#%%
K = np.linspace(0,10,100)
K = K[1:]
# tau = 0
beta = 1.5
rho = .1

plt.interactive(True)

#%% E[G_1 | K]
n = 5
C = sns.color_palette('YlOrRd',n)
# C = sns.color_palette('hsv',n)

hf, ha = plt.subplots(3,2,figsize=(7,10))#,sharey=True,sharex=True)
# hf2, ha2 = plt.subplots(1,2,figsize=(7,3.5),sharey=True,sharex=True)
# hf3, ha3 = plt.subplots(1,2,figsize=(7,3.5),sharey=True,sharex=True)

pB = .5
for i,tau in enumerate(np.hstack((0,np.logspace(-2,2,n-1)))):
    print(i,"tau={:1.2e}".format(tau))
    G1 = EG1(pB,K,beta,tau)
    k1 = kEG1(beta, tau)
    ha[0, 0].plot(K, G1, label="tau={:5.1f}".format(tau), color=C[i])
    ha[0, 0].plot(k1, EG1(pB,k1,beta,tau), '*', markersize=10, color=C[i])

    G2 = EG2(pB,K,beta,tau,rho)
    k2 = kEG2(beta,rho,pB)
    ha[1, 0].plot(K, G2, color=C[i])
    ha[1, 0].plot(k2, EG2(pB,k2,beta,tau,rho), '*', markersize=10, color=C[i])

    G3 = EG3(pB,K,beta,tau,rho)
    k3 = kEG3(beta,rho,pB)
    ha[2, 0].plot(K, G3, color=C[i])
    ha[2, 0].plot(k3, EG3(pB,k3,beta,tau,rho), '*', markersize=10, color=C[i])

tau = .1
for i, pB in enumerate(np.linspace(95, 5, n) / 100):
    print(i, "pB={:1.2f}".format(pB))
    G1 = EG1(pB, K, beta, tau)
    k1 = kEG1(beta, tau)
    ha[0, 1].plot(K, G1, label="P(B)={:1.2f}".format(pB), color=C[i])
    ha[0, 1].plot(k1, EG1(pB, k1, beta, tau), '*', markersize=10, color=C[i])

    G2 = EG2(pB, K, beta, tau, rho)
    k2 = kEG2(beta, rho, pB)
    ha[1, 1].plot(K, G2, color=C[i])
    ha[1, 1].plot(k2, EG2(pB, k2, beta, tau, rho), '*', markersize=10, color=C[i])

    G3 = EG3(pB, K, beta, tau, rho)
    k3 = kEG3(beta, rho, pB)
    ha[2, 1].plot(K, G3, color=C[i])
    ha[2, 1].plot(k3, EG3(pB, k3, beta, tau, rho), '*', markersize=10, color=C[i])

    # break
# ha1[1].plot(beta,0,'*')
# ha[2,0].set_xlabel('K \n waitingTime')
# ha[1,0].set_ylabel('E [G | K]\nexpected return')
ha[0,0].legend(frameon=False)#,fontsize='small')
ha[0,1].legend(frameon=False)#,fontsize='small')

#%% IS MY COMPLICATED K3 EQUIVALENT TO LAK 14?
beta = 1.5
rho = .2
tau = 1
pB = np.linspace(50,100,20)/100

hf_lak, ha_lak = plt.subplots(1,1,figsize=(3.5,3.5),sharey=True,sharex=True)
ha_lak.plot(pB,kLak14(beta,rho,pB),color='xkcd:dark blue',label='Lak14')
# ha_lak.plot(pB,np.full(pB.shape,1)*kEG1(beta,tau),color='xkcd:light blue',label='G1')
ha_lak.plot(pB,kEG2(beta,rho,pB),color='xkcd:bright blue',label='G2')
ha_lak.plot(pB,kEG3(beta,rho,pB),color='xkcd:sea blue',label='G3')
ha_lak.legend(frameon=False,fontsize='small')
