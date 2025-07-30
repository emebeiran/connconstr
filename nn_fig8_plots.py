#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:25:27 2023

@author: mbeiran
"""
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

#from argparse import ArgumentParser
import sys
import fun_lib as fl
from scipy.linalg import expm

fl.set_plot()


import random 
import networkx as nx

#%%
# This is for random graphs with skellam distribution (growing variance)
#from networkx.utils import powerlaw_sequence
#from scipy.stats import skellam
# N = 300
# p = 0.4
# Vars = np.array((0.1, 1.,  10.,  32.))*p*N
# Ss = []

N = 300
D = 60
np.random.seed(31)
g= 1.4
J_ = g*np.random.randn(N,N)/np.sqrt(N)

u, s, v = np.linalg.svd(J_)


lst = np.random.randint(0,N, D)
#lst = np.arange(D)
J = u[:,lst].dot(np.diag(s[lst]).dot(v[lst,:]))

u, s, v = np.linalg.svd(J)
ev, vs = np.linalg.eig(J)
ev_, vs_ = np.linalg.eig(J_)

sig_b = 0.5
b = np.random.randn(N)*sig_b

# for iV, V in enumerate(Vars):
#     sequence = skellam.rvs(  p*V, p*V, loc= p*N, size=N)
#     if np.mod(np.sum(sequence),2)==1:
#            sequence[np.random.randint(0, N)]+=1
#     sequence[sequence<=0] = np.random.randint(2, 5, size=np.sum(sequence<=0))
#     Ss.append(sequence)
#     plt.hist(sequence, alpha=0.5, bins=50)
    
    
# plt.show()
#%%
# g = 1.5

# sAs = []
# sols = []
# scores = []
# for iV, V in enumerate(Vars):
#     S2 = np.random.permutation(Ss[iV])
#     G = nx.directed_configuration_model(Ss[iV], Ss[iV])
#     #G = nx.configuration_model(Ss[-1])
#     G = nx.DiGraph(G)
#     J_ = np.array(nx.adjacency_matrix(G).todense())
iS = np.argsort(np.sum(J**2,1))
Ss_sort = np.sort( np.sum(J**2,1))
#     J = np.copy(J_)
J  = J[iS,:]
J = J[:,iS]
    
#J = (g/np.sqrt(N))*np.random.randn(N,N)*J
plt.imshow(J, cmap='bwr', vmin=-0.01, vmax=0.01)
plt.show()
    
#%%
us = np.linalg.eigvals(J)
plt.scatter(np.real(us), np.imag(us))
plt.show()
#%%
sAs = []
scores=[]
A = np.linalg.pinv(np.eye(N)-J).dot(J)
uA, sA, vA = np.linalg.svd(A)
sAs.append(sA)

Anorm =(1/np.sqrt(np.diag(A.dot(A.T)))[:,np.newaxis]) * A
#Anorm = Anorm.T

sol = Anorm.dot(vA.T).dot(np.diag(sA))
score = np.sum(sol**2, 1)/np.sum(sA**2)
scores.append(score)

plt.hist(score)    
plt.show()

plt.scatter(score, Ss_sort)
plt.show()

scores = np.array(scores)
#%%
#pall = plt.cm.inferno(np.linspace(0.2, 0.8, len(Vars)))
fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
ax = fig.add_subplot(111)

i = 0
ax.hist(scores[i,:], 50, alpha=0.8, color='C3')
#ax.set_xlim([0, 1.1*np.max(scores)])
#ax.set_yscale('log')
#plt.yscale('log')

plt.ylabel('count')
plt.xlabel('error reduction per neuron')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  

plt.savefig('Figs/F5_C_zero.pdf', transparent=True) 
plt.show()



#%%
# Randomly picking connections vs best
# analytical

def normalize_rows(M):
    nM = np.zeros_like(M)
    for i in range(np.shape(M)[0]):
        nM[i,:] = M[i,:]/np.sqrt(np.sum(M[i,:]**2))
    return(nM)

def calculate_perf(vM, svM, nM):
    sol = np.dot(vM, nM.T) #first index: correlation with first sv, second index, neuron
    corr_sv = np.zeros(N)
    for i in range(np.shape(nM)[0]):
        corr_sv[i] = np.mean((sol[:,i]*svM)**2)#/np.mean((svM**2))
    corr_sv = corr_sv/np.mean(svM**2)
    return(corr_sv)    

plt.figure()
    
   
remVs = np.zeros(N+1)
remVs[0] = 1

M = np.linalg.pinv(np.eye(N)-J).dot(J)
uM, svMo, vMo = np.linalg.svd(M, full_matrices=False)
P = np.eye(N)
M2 = np.copy(M)
nM = normalize_rows(M)
savIx = np.zeros(N, dtype=np.int32)
for i in range(N):
    M2 = P.dot(M2.T)
    M2 = M2.T
    nM = normalize_rows(M2)
    corr_sv = calculate_perf(vMo, svMo, nM)
    bestI = np.argmax(corr_sv)
    if remVs[i]>0:
        remVs[i+1] = remVs[i]-np.max(corr_sv[corr_sv!=0])
    else:
        remVs[i+1] = 0.
    #remVs[i+1] = remVs[i]-np.max(corr_sv)
    if i<1:
        print(corr_sv[bestI])
        print('max possible:'+str(np.max(svMo**2)/np.sum(svMo**2)))

    savIx[i] = np.int32(bestI)
    row = nM[[bestI,],:]
    M2 = np.delete(M2, bestI, axis=0)
    P = P - np.dot(row.T, row)

# get worst
remVsW = np.zeros(N+1)

remVsW[0] = 1
M = np.linalg.pinv(np.eye(N)-J).dot(J)
uM, svMo, vMo = np.linalg.svd(M, full_matrices=False)
P = np.eye(N)
M2 = np.copy(M)
nM = normalize_rows(M)
savIxW = np.zeros(N, dtype=np.int32)
for i in range(N):
    M2 = P.dot(M2.T)
    M2 = M2.T
    nM = normalize_rows(M2)
    corr_sv = calculate_perf(vMo, svMo, nM)
    bestI = np.argmin(corr_sv[corr_sv!=0])
    if remVsW[i]>0:
        remVsW[i+1] = remVsW[i]-np.min(corr_sv[corr_sv!=0])
    else:
        remVsW[i+1] = 0.
    savIxW[i] = np.int32(bestI)
    row = nM[[bestI,],:]
    M2 = np.delete(M2, bestI, axis=0)
    P = P - np.dot(row.T, row)
    
# get random
trials = 15
remVsR = np.zeros((N+1, trials))
for itr in range(trials):
    remVsR[0] = 1
    M = np.linalg.pinv(np.eye(N)-J).dot(J)
    uM, svMo, vMo = np.linalg.svd(M, full_matrices=False)
    P = np.eye(N)
    M2 = np.copy(M)
    nM = normalize_rows(M)
    savIxR = np.zeros(N, dtype=np.int32)
    for i in range(N):
        M2 = P.dot(M2.T)
        M2 = M2.T
        nM = normalize_rows(M2)
        corr_sv = calculate_perf(vMo, svMo, nM)
        bestI = np.random.randint(0, high=np.shape(corr_sv[corr_sv!=0]))[0]#np.argmin(corr_sv[corr_sv!=0])
        if remVsR[i, itr]>0:
            remVsR[i+1, itr] = remVsR[i, itr]-corr_sv[corr_sv!=0][bestI]
        else:
            remVsR[i+1, itr] = 0.
            break
        savIxR[i] = np.int32(bestI)
        row = nM[[bestI,],:]
        M2 = np.delete(M2, bestI, axis=0)
        P = P - np.dot(row.T, row)
#%%
fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
ax = fig.add_subplot(111)

plt.plot(np.arange(N)+1, remVs[1:],  color='C0', label='best', lw=1.5)
mm = np.mean(remVsR[1:],-1)
ss = np.std(remVsR[1:],-1)
plt.plot(np.arange(N)+1, mm,  color='k', label='random')
plt.plot(np.arange(N)+1, remVsW[1:], color='C1', label='worst', lw=1.5)
plt.fill_between(np.arange(N)+1, mm-ss, mm+ss,  color='k', alpha=0.3)

plt.xscale('log')
plt.legend(frameon=False)
plt.xlabel(r'recorded units $M$ ')
plt.ylabel(r'total error (norm.) ')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xscale('log')
plt.savefig('Figs/F5_C_one.pdf', transparent=True) 
plt.show()

#%%
fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
ax = fig.add_subplot(111)

plt.plot(np.arange(N)+1, remVs[1:]/(1.-(np.arange(N)-1)/(N)),  color='C0', label='best', lw=1.5)
mm = np.mean(remVsR[1:]/(1.-(np.arange(N)-1)/(N))[:,None],-1)
ss = np.std(remVsR[1:]/(1.-(np.arange(N)-1)/(N))[:,None],-1)
plt.plot(np.arange(N)+1, mm,  color='k', label='random')
plt.plot(np.arange(N)+1, remVsW[1:]/(1.-(np.arange(N)-1)/(N)), color='C1', label='worst', lw=1.5)
plt.fill_between(np.arange(N)+1, mm-ss, mm+ss,  color='k', alpha=0.3)

plt.xscale('log')
plt.legend(frameon=False)
plt.xlabel(r'recorded units $M$ ')
plt.ylabel(r'error unrec. act. (norm.) ')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xscale('log')
ax.set_yticks([0, 0.5, 1.])
plt.savefig('Figs/F5_C_unrec.pdf', transparent=True) 
plt.show()

#%%
fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
ax = fig.add_subplot(111)

plt.plot(np.arange(N)+1, 0*remVs[1:]/(1.-(np.arange(N)-1)/(N)),  color='C0', label='best', lw=1.5)
mm = np.mean(0*remVsR[1:]/(1.-(np.arange(N)-1)/(N))[:,None],-1)
ss = np.std(0*remVsR[1:]/(1.-(np.arange(N)-1)/(N))[:,None],-1)
plt.plot(np.arange(N)+1, mm,  color='k', label='random')
plt.plot(np.arange(N)+1, 0*remVsW[1:]/(1.-(np.arange(N)-1)/(N)), color='C1', label='worst', lw=1.5)
plt.fill_between(np.arange(N)+1, mm-ss, mm+ss,  color='k', alpha=0.3)

plt.xscale('log')
plt.legend(frameon=False)
plt.xlabel(r'recorded units $M$ ')
plt.ylabel(r'error rec. act. (norm.) ')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xscale('log')
ax.set_yticks([0, 0.5, 1.])
plt.savefig('Figs/F5_C_rec.pdf', transparent=True) 
plt.show()

#%%


def phi(x):
    return(x)

def get_fp_num(J, g, b, phi):
    Jn = np.dot(J, np.diag(g))
    x = 0
    i = 0
    imax = 1000
    err = 1
    dt = 0.1
    while i<imax and err>1e-9:
        x_new = x+dt*(-x+Jn.dot(phi(x+b)))
        err = np.mean((x-x_new)**2)
        i = i +1
        x = x_new

    return(x_new)  


#%%

try:
    AA = np.load('Data/Figure8/loss_linear_sequence100.npz')
    loss0 = AA['arr_0']
    loss_b0 = AA['arr_1']
    loss0R = AA['arr_2']
    loss_b0R = AA['arr_3']
    loss0W = AA['arr_4']
    loss_b0W = AA['arr_5']
    
except:
    np.random.seed(21)
    trials = 1
    tars = 50
    T = 100
    
    loss0 = np.zeros((trials, tars, N))
    loss_b0 = np.zeros((trials, tars, N))
    
    loss0R = np.zeros((trials, tars, N))
    loss_b0R = np.zeros((trials, tars, N))
    
    loss0W = np.zeros((trials, tars, N))
    loss_b0W = np.zeros((trials, tars, N))
    
    s_b = 4.5
    gt = np.ones(N)
    M = np.linalg.pinv(np.eye(N)-J).dot(J)
    
    for ta in range(tars):
        if np.mod(ta,10)==0:
            print('tar')
            print(ta)
        bt = s_b*np.random.randn(N)
        xt = get_fp_num(J, gt, bt, phi)
        xt2 = np.dot(M, bt)
        for tr in range(trials):
            iN = 0
            bt0 = s_b*np.random.randn(N)
            e0 = bt0-bt
            x0 = np.dot(M, e0)
            e00 = np.mean(e0**2)
            e00_ = np.copy(e0)
            x00 = np.mean(x0**2)
            x00_ = np.copy(x0)
    
            Mc = np.copy(M)
            P = np.eye(N)
            for iN in range(N):
                x0 = np.dot(M, e0)
                loss_b0[tr, ta, iN] = np.mean(e0**2)/e00
                loss0[tr, ta, iN] = np.mean(x0**2)/x00
                if loss0[tr,ta,iN]<1e-8:
                    loss0[tr, ta, iN+1:] = loss0[tr, ta, iN]
                    loss_b0[tr, ta, iN+1:] = loss_b0[tr, ta, iN]
                    break
                row = Mc[[savIx[iN],],:] #here
                Proj = row.T.dot(row)/(row.dot(row.T))
                P = P-Proj
    
                ef = e0-Proj.dot(e0)
                xf = np.dot(M, ef)  
                Mc = np.delete(Mc, savIx[iN], axis=0)
                Mc = np.dot(P, Mc.T).T
                e0 = ef
           
    #% random
    

    for ta in range(tars):
        savIxR = np.random.permutation(np.arange(N))
        for i in range(N):
            savIxR[i] = np.random.randint(0, high = N-i, size=1)   
        if np.mod(ta,10)==0:
            print('tar')
            print(ta)
        bt = s_b*np.random.randn(N)
        xt = get_fp_num(J, gt, bt, phi)
        xt2 = np.dot(M, bt)
        for tr in range(trials):
            iN = 0
            bt0 = s_b*np.random.randn(N)
            e0 = bt0-bt
            x0 = np.dot(M, e0)
            e00 = np.mean(e0**2)
            x00 = np.mean(x0**2)
    
            Mc = np.copy(M)
            P = np.eye(N)
            for iN in range(N):
                x0 = np.dot(M, e0)
                loss_b0R[tr, ta, iN] = np.mean(e0**2)/e00
                loss0R[tr, ta, iN] = np.mean(x0**2)/x00
                if loss0[tr,ta,iN]<1e-8:
                    loss0R[tr, ta, iN+1:] = loss0R[tr, ta, iN]
                    loss_b0R[tr, ta, iN+1:] = loss_b0R[tr, ta, iN]
                    break
                row = Mc[[savIxR[iN],],:] #here
                Proj = row.T.dot(row)/(row.dot(row.T))
                P = P-Proj
    
                ef = e0-Proj.dot(e0)
                xf = np.dot(M, ef)  
                Mc = np.delete(Mc, savIxR[iN], axis=0)
                Mc = np.dot(P, Mc.T).T
                e0 = ef
                
    #% worst
    
    for ta in range(tars):
        if np.mod(ta,10)==0:
            print('tar')
            print(ta)
        bt = s_b*np.random.randn(N)
        xt = get_fp_num(J, gt, bt, phi)
        xt2 = np.dot(M, bt)
        for tr in range(trials):
            iN = 0
            bt0 = s_b*np.random.randn(N)
            e0 = bt0-bt
            x0 = np.dot(M, e0)
            e00 = np.mean(e0**2)
            x00 = np.mean(x0**2)
    
            Mc = np.copy(M)
            P = np.eye(N)
            for iN in range(N):
                x0 = np.dot(M, e0)
                loss_b0W[tr, ta, iN] = np.mean(e0**2)/e00
                loss0W[tr, ta, iN] = np.mean(x0**2)/x00
                if loss0W[tr,ta,iN]<1e-8:
                    loss0W[tr, ta, iN+1:] = loss0W[tr, ta, iN]
                    loss_b0W[tr, ta, iN+1:] = loss_b0W[tr, ta, iN]
                    break
                row = Mc[[savIxW[iN],],:] #here
                Proj = row.T.dot(row)/(row.dot(row.T))
                P = P-Proj
    
                ef = e0-Proj.dot(e0)
                xf = np.dot(M, ef)  
                Mc = np.delete(Mc, savIxW[iN], axis=0)
                Mc = np.dot(P, Mc.T).T
                e0 = ef
    
    #%
    np.savez('Data/Figure8/oss_linear_sequence100.npz', loss0, loss_b0, loss0R, loss_b0R, loss0W, loss_b0W)

# #%%
# Loss0 = np.reshape(loss0, np.array((loss0.shape[0]*loss0.shape[1], loss0.shape[2])))
# mm = np.mean(Loss0, 0)
# ss = np.std(Loss0, 0)/np.sqrt(Loss0.shape[0])
# plt.scatter(np.arange(N-1)+1, mm[1:])
# plt.errorbar(np.arange(N-1)+1, mm[1:], yerr=ss[1:])
# Loss0 = np.reshape(loss0R, np.array((loss0R.shape[0]*loss0R.shape[1], loss0R.shape[2])))
# mm = np.mean(Loss0, 0)
# ss = np.std(Loss0, 0)/np.sqrt(Loss0.shape[0])
# plt.scatter(np.arange(N-1)+1, mm[1:])
# plt.errorbar(np.arange(N-1)+1, mm[1:], yerr=ss[1:])
# plt.xscale('log')
# plt.show()

# #%%
# plt.plot(np.arange(N), np.nanmean(loss_b0,0).T)
# plt.xscale('log')
# plt.show()


#%%
fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
ax = fig.add_subplot(111)

plt.plot(np.arange(N)+1, remVs[1:]/(1.-(np.arange(N)-1)/(N)),  color='C0', label='best', lw=1.5)
mm = np.mean(remVsR[1:]/(1.-(np.arange(N)-1)/(N))[:,None],-1)
ss = np.std(remVsR[1:]/(1.-(np.arange(N)-1)/(N))[:,None],-1)
plt.plot(np.arange(N)+1, mm,  color='k', label='random')
plt.plot(np.arange(N)+1, remVsW[1:]/(1.-(np.arange(N)-1)/(N)), color='C1', label='worst', lw=1.5)
plt.fill_between(np.arange(N)+1, mm-ss, mm+ss,  color='k', alpha=0.3)

Loss0 = np.reshape(loss0, np.array((loss0.shape[0]*loss0.shape[1], loss0.shape[2])))
mm = np.mean(Loss0, 0)
ss = np.std(Loss0, 0)/np.sqrt(Loss0.shape[0])
plt.scatter(np.arange(N-1)+1, mm[1:])
plt.errorbar(np.arange(N-1)+1, mm[1:], yerr=ss[1:], c='C0', fmt='o')
#plt.errorbar(np.arange(N-1)+1, mm[1:], yerr=ss[1:])

Loss0 = np.reshape(loss0R, np.array((loss0R.shape[0]*loss0R.shape[1], loss0R.shape[2])))
mm = np.mean(Loss0, 0)
ss = np.std(Loss0, 0)/np.sqrt(Loss0.shape[0])
plt.scatter(np.arange(N-1)+1, mm[1:], c='k')
plt.errorbar(np.arange(N-1)+1, mm[1:], yerr=ss[1:], c='k', fmt='o')

Loss0 = np.reshape(loss0W, np.array((loss0W.shape[0]*loss0W.shape[1], loss0W.shape[2])))
mm = np.mean(Loss0, 0)
ss = np.std(Loss0, 0)/np.sqrt(Loss0.shape[0])
plt.scatter(np.arange(N-1)+1, mm[1:], c='C1')
plt.errorbar(np.arange(N-1)+1, mm[1:], yerr=ss[1:], c='C1', fmt='o')
plt.xscale('log')

plt.xscale('log')
plt.legend(frameon=False)
plt.xlabel(r'recorded units $M$ ')
plt.ylabel(r'error unrec. act. (norm.) ')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xscale('log')
ax.set_yticks([0, 0.5, 1.])
ax.set_ylim([0, 1.0])
plt.savefig('Figs/F5_C_unrec_sim.pdf', transparent=True) 
plt.show()