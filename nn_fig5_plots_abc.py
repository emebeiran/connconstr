#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:58:28 2022

@author: mbeiran
"""

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import fun_lib2 as fs



#%%
def relu(x):
    return np.log(1+np.exp(x))

#%%    
fs.set_plot()

#%%
input_size = 2

netN = 0

AA = np.load('Data/Figure5/ashokF_softplus.npz')#np.load('/users/mbeiran/Downloads/larval_locomotion-master-2/RNN/ashok_'+str(netN)+'.npz')
Jpm = AA['arr_0']
Jpp = AA['arr_1']
bm = AA['arr_2']
bp = AA['arr_3']
taum = AA['arr_4']
taup = AA['arr_5']
gm = AA['arr_6']
gp = AA['arr_7']
wsp = AA['arr_8']
p0 = AA['arr_9']
m0 = AA['arr_10']
BB = np.load('Data/Figure5/ashok_s.npz')
s = BB['arr_0']
NT = np.sum(np.shape(Jpm))
Np = np.shape(Jpm)[0]
Nm = np.shape(Jpm)[1]
J = np.zeros((NT, NT))
J[0:Np,0:Np] = Jpp#*Jpp_mask #First P
J[0:Np,Np:] = Jpm#*Jpm_mask #Then m

tMs = taup*np.ones(NT)
hidden_size = NT
output_size = Nm
wout = np.zeros((NT, output_size))
for i in range(output_size):
    wout[-i,i] = 1
x0 = 0*np.hstack((p0[0,0,:], m0[0,0,:])) #selecting only the initial condition for forward
wI = np.zeros((input_size, NT))
wI[:,0:Np] = wsp
gf = np.hstack((gp, gm))
bf = np.hstack((bp, bm))


#%%
sig_b = np.std(bf)
sig_g = np.std(gf)

    
#%%
seed = 21
sd = 0.001
n_ep = 210
np.random.seed(seed)
bt = sig_b*np.random.randn(NT)+np.mean(bf)


dtype = torch.FloatTensor  
wi_init = torch.from_numpy(wI).type(dtype)
wo_init = torch.from_numpy(wout[:,np.newaxis]).type(dtype)

wrec_init = torch.from_numpy(J).type(dtype)
b_init_True= torch.from_numpy(bf[:,np.newaxis]).type(dtype)


b2_mask = np.arange(len(bf))
b2_mask[0:Np] = np.random.permutation(b2_mask[0:Np])
b_init_Ran= torch.from_numpy(bf[b2_mask,np.newaxis]).type(dtype)
b_init= torch.from_numpy(bf[:,np.newaxis]).type(dtype)



H0= torch.from_numpy(x0).type(dtype)
#%%
trials = 2
dta = 0.05
Ntim = np.shape(s)[0] 
taus = np.linspace(0, 1, Ntim)
taus = taus*dta
Nt = len(taus)
alpha = dta

#%%

s2 = np.copy(s)
s2[90:,:,:]= 0
input_train = np.zeros((trials, Nt, 2))
output_train = np.zeros((trials, Nt, output_size))
output_train2 = np.zeros((trials, Nt, NT))
output_trainNoInp = np.zeros((trials, Nt, NT))

bf_random = np.random.permutation(bf)
for tr in range(trials):
    input_train[tr,:,:] = s2[:,tr,:]
    
    print(tr)
    
    xB = np.zeros((len(taus), NT))
    rB = np.zeros((len(taus)))
    
    xB[0,:] = x0
    output_train[tr,0,:] = wout.T.dot(relu(x0))
    
    
    
    output_train2[tr,0,:]= relu(xB[0,:])
    output_trainNoInp[tr,0,:]= relu(xB[0,:])
    
    
    #     M[it+1,:,:] = M[it,:,:]*(1-dt/s_taum)+(dt/s_taum)*(s_gm*np.dot(rlu(P[it,:,:]), s_Jpm*sJpm_mask)+s_bm)
    #     P[it+1,:,:] = P[it,:,:]*(1-dt/s_taup)+(dt/s_taup)*(s_gp*np.dot(rlu(P[it,:,:]), s_Jpp*sJpp_mask)+s_bp+np.dot(s_s[np.min((it+1, 119)),:,:],s_wsp))

    for it, ta in enumerate(taus[:-1]):
        
        xB[it+1,:]=xB[it,:]+(dta/tMs)*(-xB[it,:]+np.diag(gf).dot(J.T.dot(relu(xB[it,:])))   + bf + s2[it,tr,:].dot(wI))
        output_train[tr,it+1,:] = wout.T.dot(relu(x0))
        output_train2[tr,it+1,:] = relu(xB[it+1,:])
        
    for it, ta in enumerate(taus[:-1]):
        
        xB[it+1,:]=xB[it,:]+(dta/tMs)*(-xB[it,:]+np.diag(gf*0.).dot(J.T.dot(relu(xB[it,:])))   + bf_random + s2[it,tr,:].dot(wI))
        output_trainNoInp[tr,it+1,:] = relu(xB[it+1,:])
        
    
#%%
perms = 100
loss0_r2 = np.zeros(perms)
loss0 = np.zeros(perms)
for ipp in range(perms):
    output_st = output_train2[:,:,np.random.permutation(np.arange(NT))] 
    # otG = np.vstack((output_st[0,:,0:Np], output_st[1,:,0:Np]))
    # ot2 = np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    # CC = np.corrcoef(otG.T, ot2.T)
    
    otG = output_st[0,:,0:Np]#np.vstack((output_train_G[0,:,0:Np], output_train_G[1,:,0:Np]))
    ot2 = output_train2[0,:,0:Np]#np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    CC1 = np.corrcoef(otG.T, ot2.T)
    otG = output_st[1,:,0:Np]#np.vstack((output_train_G[0,:,0:Np], output_train_G[1,:,0:Np]))
    ot2 = output_train2[1,:,0:Np]#np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    CC2 = np.corrcoef(otG.T, ot2.T)
    CC = 0.5*(CC1+CC2)
    
    lossr2 = np.diag(CC[0:Np,Np:])     
    loss0_r2[ipp] = 1-np.mean(np.abs(lossr2))
    loss = (output_st[:,:,0:Np]-output_train2[:,:,0:Np])**2
    loss0[ipp] = np.mean(np.mean(np.mean(loss,0),0))
    
#%%


percs = np.array((1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 88, 96, 112, 128, 144, 178,))

#percs = #np.array((1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 144, ))#178,))
#np.array((1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 144, 178,)) #Softplus2
#np.array((1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 128, 144, 178,))

inf_sort = True

wo_init2_ = np.eye(hidden_size)
losses_ = []
eG_ = []

losses = []
eG = []

#g_Ran = relu(0.8*np.std(gf[0:Np])*np.random.randn(NT,1)+0.8*np.mean(gf[0:Np]))#gf[:,np.newaxis] #sig_g*np.random.randn(N,1)+1 #
g_Ran = np.abs(0.5*np.std(gf[0:Np])*np.random.randn(NT,1)+0.5*np.mean(gf[0:Np]))#gf[:,np.newaxis] #sig_g*np.random.randn(N,1)+1 #

g_init_Ran = torch.from_numpy(g_Ran).type(dtype)
g_init_True = torch.from_numpy(gf[:,np.newaxis]).type(dtype)

tM = torch.from_numpy(tMs).type(dtype)
n_ep = 8000#500#400
rep_n_ep = 1#5#5


#%%
lrs = [ 0.003, ] #ne
lr = lrs[0]
save_neurs = 100

count = 0
used_percs = percs
e=False
ics = 20
stri = 'Data/Figure5/'
N = NT

dor2 = True

try:
    aa = np.load(stri+'data_fig5_abc.npz')
    losses = aa['arr_0']
    losses_read= aa['arr_1']
    losses_unk = aa['arr_2']
    losses_kno = aa['arr_3']
    meG = aa['arr_4']
    #sameG = aa['arr_5']
except:
    print('loading')
    for ilr, lr in enumerate(lrs):
        for ic in range(ics):
            for ip in range(len(used_percs)-1):
    
                try:
                    AA = np.load(stri+'taskSoftPlusH1_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
                except:
                    print('hey: '+str(ic)+', ip:'+str(ip))
                    AA = np.load(stri+'taskSoftPlusH1_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(1)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
                
                    #    if ic==10 and ip==9:
                #        AA = np.load(stri+'taskSoftPlus4_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(8)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
                losses_ = AA['arr_0']
                losses_read_ = AA['arr_1']
                if dor2:
                    losses_kno_ = AA['arr_8']
                    losses_unk_ = AA['arr_9']
                else:
                    losses_kno_ = AA['arr_2']
                    losses_unk_ = AA['arr_3']
        
                
                e_g = AA['arr_4']

                
                
                if count==0:
                    losses = np.zeros((len(losses_), len(used_percs), ics, len(lrs)))
                    losses_read = np.zeros((len(losses_read_), len(used_percs), (ics), len(lrs)))
                    losses_unk = np.zeros((len(losses_unk_), len(used_percs), (ics), len(lrs)))
                    losses_kno = np.zeros((len(losses_kno_), len(used_percs), (ics), len(lrs)))
                    meG = np.zeros((np.shape(e_g)[1], len(used_percs), (ics), len(lrs)))
                    sameG = np.zeros((save_neurs, np.shape(e_g)[1], len(used_percs), (ics), len(lrs)))
                    count +=1
                    
                if e:
                    losses[:,ip, ic, ilr] = np.nan
                    losses_read[:,ip, ic, ilr] = np.nan
                    losses_unk[:,ip, ic, ilr] = np.nan
                    losses_kno[:,ip, ic, ilr] = np.nan
                    meG[:,ip, ic, ilr] = np.nan
                    #i_eG[:,:,ip, ic, ilr] = np.nan
                    #f_eG[:,:,ip, ic, ilr] = np.nan
                    #sameG[:,ip, ic, ilr] = np.nan
                    
                else:
                    losses[:,ip, ic, ilr] = losses_
                    losses_read[:,ip, ic, ilr] = losses_read_
                    losses_unk[:,ip, ic, ilr] = losses_unk_
                    losses_kno[:,ip, ic, ilr] = losses_kno_
                    meG[:,ip, ic, ilr] = np.sqrt(np.mean(e_g**2,axis=(0)))
                    #i_eG[:,:,ip, ic, ilr] = e_g[:,:,0]
                    #f_eG[:,:,ip, ic, ilr] = e_g[:,:,-1]
    
                    sameG[0:save_neurs,:,ip, ic, ilr] = e_g[0:save_neurs,:]
    np.savez('data_fig5_abc.npz', losses, losses_read, losses_unk, losses_kno, meG)
                
#%%
losses[losses==0] = np.nan
losses_read[losses_read==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan


#%%
Lo = losses_kno[:,:,:,0]
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(212)
# for ip, pi in enumerate(percs[:-1]):
#     mm = np.nanmean(Lo[:,ip,:],-1)/np.mean(Lo[0,ip,:], -1)
#     ss = np.nanstd(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
#     xx = np.arange(len(mm))*10
#     plt.plot(xx, mm, c=palette[ip])
#     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)

# ax.set_ylabel('norm.')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_yscale('log')
# ax.set_yticks([0.001, 0.1,])

#ax = fig.add_subplot(211)
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))
    xx = np.arange(len(mm))*10
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('$loss_{known}$')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_yscale('log')
#ax.set_yticks([0.001, 0.1,])
plt.savefig('Figs/AshokEI_knownLoss.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure(figsize=[1.5*3.5, 1.5*2.7])
ax = fig.add_subplot(212)
for ip, pi in enumerate(percs):
    mm = np.nanmean(Lo[:,ip,:],-1)/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    xx = np.arange(len(mm))*10
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('norm.')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_yscale('log')
#ax.set_yticks([0.01, 0.1,])
ax = fig.add_subplot(211)
for ip, pi in enumerate(percs):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))
    xx = np.arange(len(mm))*10
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('$loss_{unknown}$')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')     
#ax.set_yscale('log') 
#ax.set_yticks([0.01, 0.1,])
plt.savefig('Figs/AshokEI_unknownLoss.pdf')

#%%
trys = 100
loss_un = 0
for tri in range(trys):
    rnd_list = np.random.permutation(np.arange(Np))

    otG = np.vstack((output_train2[0,:,rnd_list[0:Np]].T, output_train2[1,:,rnd_list[0:Np]].T))
    ot2 = np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    CC = np.corrcoef(otG.T, ot2.T)
    
    # otG = output_train2[0,1:,rnd_list[0:Np]].T#np.vstack((output_train_G[0,:,0:Np], output_train_G[1,:,0:Np]))
    # ot2 = output_train2[0,1:,0:Np]#np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    # CC1 = np.corrcoef(otG.T, ot2.T)
    # #otG = output_train_G[1,1:,0:Np]#np.vstack((output_train_G[0,:,0:Np], output_train_G[1,:,0:Np]))
    # #ot2 = output_train2[1,1:,0:Np]#np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    # CC2 = np.corrcoef(otG.T, ot2.T)
    # CC = 0.5*(CC1+CC2)
    lossr2 = np.diag(CC[0:Np,Np:])
    if dor2:       
        loss_un += (1-np.mean(np.abs(lossr2)))/trys
print(loss_un)
#%%
if not dor2:
    loss_un  = np.mean(loss0)
    
loss_unNoInp = 0
for tri in range(trys):
    # otG = np.vstack((output_trainNoInp[0,:,0:Np], output_trainNoInp[1,:,0:Np]))
    # ot2 = np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    # CC = np.corrcoef(otG.T, ot2.T)
    otG = output_trainNoInp[0,1:,0:Np]#np.vstack((output_train_G[0,:,0:Np], output_train_G[1,:,0:Np]))
    ot2 = output_train2[0,1:,0:Np]#np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    CC1 = np.corrcoef(otG.T, ot2.T)
    #otG = output_train_G[1,1:,0:Np]#np.vstack((output_train_G[0,:,0:Np], output_train_G[1,:,0:Np]))
    #ot2 = output_train2[1,1:,0:Np]#np.vstack((output_train2[0,:,0:Np], output_train2[1,:,0:Np]))
    CC2 = np.corrcoef(otG.T, ot2.T)
    CC = 0.5*(CC1+CC2)
    lossr2 = np.diag(CC[0:Np,Np:])
    if dor2:       
        loss_unNoInp += (1-np.mean(lossr2))/trys


#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs[:-2]):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    m_all.append(mm[-1])
plt.plot(percs[:-2], m_all, c='k', lw=0.5)
ax.set_ylabel(r'$1-|\rho|$  (unrec. activity)')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
plt.plot(percs, loss_un*np.ones_like(percs), c='k', alpha=0.3, lw=4)
#plt.ylim([0, 0.11])
#plt.yscale('log')
plt.savefig('Figs/f_neuraldata_unknownLoss.pdf')
# # plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs[:-2]):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    m_all.append(mm[-1])
    
    if ip==0:
        pi=0.5
        plt.scatter(pi, mm[0], s=40, color='k', edgecolor='k', zorder=4)
        plt.plot([pi, pi], [mm[0]-ss[0], mm[0]+ss[0]], color='k')
Percs = [0.5, percs[-1]]
#plt.plot(Percs, (loss_un)*np.ones_like(Percs), c='k', alpha=0.3, lw=4)
# if dor2:
#     plt.yticks([0.03, 0.05, 0.07, 0.09, 0.13])
#     plt.ylim([0.01, 0.14])


plt.plot(percs[:-2], m_all, c='k', lw=0.5)
ax.set_ylabel(r'1-$\left[\rho\right]$ (unrecorded act.) ' ) #error unrec. activity')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
#plt.plot(percs, loss_un*np.ones_like(percs), c='k', alpha=0.3, lw=4)
#plt.ylim([0, 0.11])
#plt.yscale('log')
plt.savefig('Figs/f_neuraldata_unknownLoss2.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs[:-2]):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    m_all.append(mm[-1])
    
    if ip==0:
        pi=0.5
        plt.scatter(pi, mm[0], s=40, color='k', edgecolor='k', zorder=4)
        plt.plot([pi, pi], [mm[0]-ss[0], mm[0]+ss[0]], color='k')
Percs = [0.5, percs[-1]]
plt.plot(Percs, (loss_unNoInp)*np.ones_like(Percs), '--', c='k', alpha=0.3, lw=4)
plt.plot(Percs, (loss_un)*np.ones_like(Percs), c='k', alpha=0.3, lw=4)

# if dor2:
#     plt.yticks([0.03, 0.05, 0.07, 0.09, 0.13])
#     plt.ylim([0.01, 0.14])


plt.plot(percs[:-2], m_all, c='k', lw=0.5)
ax.set_ylabel(r'1-|$\rho$| (unrecorded act.) ' ) #error unrec. activity')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
#plt.plot(percs, loss_un*np.ones_like(percs), c='k', alpha=0.3, lw=4)
#plt.ylim([0, 0.11])
#plt.yscale('log')
plt.savefig('Figs/f_neuraldata_unknownLoss2_noJ.pdf')

#%%
Lo = losses_kno[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs[:-2]):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)#/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    #plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    plt.scatter(pi*np.ones(len(Lo[-1,ip,:])), Lo[-1,ip,:], s=20, color=palette[ip], zorder=4)
    
    m_all.append(mm[-1])
    
    if ip==0:
        pi=0.5
        plt.scatter(pi, mm[0], s=40, color='k', edgecolor='k', zorder=4)
        plt.plot([pi, pi], [mm[0]-ss[0], mm[0]+ss[0]], color='k')
Percs = [0.5, percs[-1]]
plt.plot(Percs, (loss_un)*np.ones_like(Percs), c='k', alpha=0.3, lw=4)
# if dor2:
#     plt.yticks([0.03, 0.05, 0.07, 0.09, 0.13])
#     plt.ylim([0.01, 0.14])


plt.plot(percs[:-2], m_all, c='k', lw=0.5)
ax.set_ylabel(r' error unrec. activity')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
ax.set_yscale('log')

#plt.plot(percs, loss_un*np.ones_like(percs), c='k', alpha=0.3, lw=4)
#plt.ylim([0, 0.11])
#plt.yscale('log')
plt.savefig('Figs/f_neuraldata_knownLoss2.pdf')
#%%
Lo = losses_kno[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs[:-2]):
    mm = np.median(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.std(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    m_all.append(mm[-1])
plt.plot(percs[:-2], m_all, c='k', lw=0.5)
ax.set_ylabel('$ loss_{known}(T)$')
ax.set_xlabel('known $N$')
ax.set_ylabel('$loss_{known}$')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
plt.ylim([-0.02, 0.11])

#plt.yscale('log')
plt.savefig('Figs/AshokEI_knownLoss_summ.pdf')
# # plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')


#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    xx[xx>2*loss_un] = np.nan
    mm = np.nanmedian(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
    
    if ip==0:
        xx = Lo[0,ip,:]
        pi=0.5
        plt.scatter(pi, mm[0], s=40, color='k', edgecolor='k', zorder=4)
        plt.plot([pi, pi], [mm[0]-ss[0], mm[0]+ss[0]], color='k')
        plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
        print(xx)

plt.plot(percs, m_all, c='k', lw=0.5)
plt.plot(Percs, (loss_un)*np.ones_like(Percs), c='k', alpha=0.3, lw=4)
plt.plot(Percs, (loss_unNoInp)*np.ones_like(Percs), c='k', alpha=0.3, lw=4)

ax.set_ylabel('$ loss_{unknown}(T)$')
ax.set_xlabel('known $N$')
ax.set_ylabel('$loss_{unknown}$')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
#plt.yscale('log')
plt.savefig('Figs/AshokEI_unknownLoss_summ2.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.mean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.std(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    #plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)
ax.set_ylabel('$ loss_{unknown}(T)$')
ax.set_xlabel('known $N$')
ax.set_ylabel('$loss_{unknown}$')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
plt.yscale('log')
plt.savefig('Figs/AshokEI_unknownLoss_summ3.pdf')
# plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')



#%%
Lo = meG[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):

    mm = np.mean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.std(Lo[:,ip,:], -1)/np.sqrt(len(percs))
    xx = 10*np.arange(len(mm))
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel(r' error gain $\theta$')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('epochs')



plt.savefig('Figs/AshokEI_gains.pdf', transparent=True)
#%%
Lo = meG[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    mm = np.mean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)
ax.set_ylabel('error g (T)')
ax.set_xlabel('known $N$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
plt.savefig('Figs/SGD_Adam_trial_percs_paramError_final.pdf')


#%%

PERCS = [2, 24, 48, ]#percs = np.array((1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 144, ))
#percs = np.array((1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 128, 144, 178,))
normed = True
for Perc in PERCS:
    print(Perc)
    AA = np.load('Data/Figure5/ashokF_softplus.npz')#np.load('/users/mbeiran/Downloads/larval_locomotion-master-2/RNN/ashok_'+str(netN)+'.npz')
    gf = np.hstack((gp, gm))
    bf = np.hstack((bp, bm))
    
    ip = np.argmin(np.abs(percs-Perc))
    ic = 1#16   
    print((stri+'taskSoftPlusH1_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz'))
    AA = np.load(stri+'taskSoftPlusH1_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
    losses_ = AA['arr_0']
    losses_read_ = AA['arr_1']
    if dor2:
        losses_kno_ = AA['arr_8']
        losses_unk_ = AA['arr_9']
    else:
        losses_kno_ = AA['arr_2']
        losses_unk_ = AA['arr_3']
    print(losses_unk_[-1])
    e_g = AA['arr_4']
    e_b = AA['arr_6']
    
    gS_0 = AA['arr_5']+e_g[:,0]
    bS_0 = AA['arr_7']+e_b[:,0]
    
    
    gS = AA['arr_5']+e_g[:,-1]
    bS = AA['arr_7']+e_b[:,-1]
    
    g_Teacher = torch.from_numpy(gf[:,np.newaxis]).type(dtype)
    b_Teacher = torch.from_numpy(bf[:,np.newaxis]).type(dtype)
    g_St = torch.from_numpy(gS[:,np.newaxis]).type(dtype)
    b_St = torch.from_numpy(bS[:,np.newaxis]).type(dtype)
    g_St0 = torch.from_numpy(gS_0[:,np.newaxis]).type(dtype)
    b_St0 = torch.from_numpy(bS_0[:,np.newaxis]).type(dtype)
    tM = torch.from_numpy(tMs).type(dtype)
    n_ep = 8000#500#400
    rep_n_ep = 1#5#5
    
    #%
    al = Np#percs[ial]
    np.random.seed(ic+10)
    mg = np.mean(gf)
    sg = np.std(gf)
    g_Ran = np.random.permutation(gf)#0.7*mg+np.random.randn(len(gf))*sg*0.7#0.2*np.random.permutation(gf)#0.5*(np.random.permutation(gf)-np.mean(gf))+0.5*np.mean(gf)#np.abs(0.5*np.std(gf[0:Np])*np.random.randn(NT,1)+0.5*np.mean(gf[0:Np]))#gf[:,np.newaxis] #sig_g*np.random.randn(N,1)+1 #
    # g_Ran[g_Ran<0.2] = 0.2
    # g_Ran[g_Ran>5] = 5
    #g_Ran = np.abs(0.5*np.std(gf[0:Np])*np.random.randn(NT,1)+0.5*np.mean(gf[0:Np]))#gf[:,np.newaxis] #sig_g*np.random.randn(N,1)+1 #
    g_init_Ran = torch.from_numpy(g_Ran[:,np.newaxis]).type(dtype)
    g_init_True = torch.from_numpy(gf[:,np.newaxis]).type(dtype)
    
    tM = torch.from_numpy(tMs).type(dtype)
    n_ep = 5000#500#400
    rep_n_ep = 1#5#5
    
    
    #np.array((0.0, 1/600, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5,  0.75, 0.9, 1.0))
    Ntot = int(al)
    rn_ar = np.random.permutation(np.arange(Np))

    #%
    wo_init2 = torch.from_numpy(wo_init2_).type(dtype)
    input_Train = torch.from_numpy(input_train).type(dtype)
    
    lrs = [0.003,]#[ 0.005, ] #new
    lr = lrs[0]
    ial=len(percs)-1 
    
    Ntot = int(al)
    
    NetJ_Teacher = fs.RNN(input_size, hidden_size, al, wi_init, wo_init2[:,0:Np], wrec_init, b_Teacher, g_Teacher, tM, h0_init=H0,
                              train_b = True, train_g=True, train_conn = False, train_wout = False, noise_std=0.00, alpha=alpha, softplus=True)
    NetJ_Student = fs.RNN(input_size, hidden_size, al, wi_init, wo_init2[:,0:Np], wrec_init, b_St, g_St, tM, h0_init=H0,
                              train_b = True, train_g=True, train_conn = False, train_wout = False, noise_std=0.00, alpha=alpha, softplus=True)
    NetJ_Student0 = fs.RNN(input_size, hidden_size, al, wi_init, wo_init2[:,0:Np], wrec_init, b_St0, g_St0, tM, h0_init=H0,
                              train_b = True, train_g=True, train_conn = False, train_wout = False, noise_std=0.00, alpha=alpha, softplus=True)
    
    Out_Te = NetJ_Teacher(input_Train)
    out_Te = Out_Te.detach().numpy()
    
    Out_St = NetJ_Student(input_Train)
    out_St = Out_St.detach().numpy()
    Out_St0 = NetJ_Student0(input_Train)
    out_St0 = Out_St0.detach().numpy()
    
    sel = 3
    
    otG = np.vstack((out_St[0,:,0:Np], out_St[1,:,0:Np]))
    otG0 = np.vstack((out_St0[0,:,0:Np], out_St0[1,:,0:Np]))
    
    ot2 = np.vstack((out_Te[0,:,0:Np], out_Te[1,:,0:Np]))
    CC = np.corrcoef(otG.T, ot2.T)
    lossr2 = np.diag(CC[0:Np,Np:])    
    
    CC = np.corrcoef(otG0.T, ot2.T)
    lossr20 = np.diag(CC[0:Np,Np:])    
    
    if dor2:
        lk  = 1-np.mean(lossr2[rn_ar[0:percs[ip]]])
        lu = 1-np.mean(lossr2[rn_ar[percs[ip]:]])
    else:
        loss__ = (ot2[:,0:Np]-otG[:,0:Np])**2
        
        lk  = np.mean(np.mean(np.mean(loss__[rn_ar[0:percs[ip]]],0),0)) #1-np.mean(lossr2)
        lu = np.mean(np.mean(np.mean(loss__[rn_ar[percs[ip]:]],0),0)) #1-np.mean(lossr2[percs[ip]:])


    rng_kno = rn_ar[0:Perc][0:sel]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    tt = np.arange(len(out_Te[0,:,0]))*0.1/5
    RNG_KNO = np.random.permutation(rng_kno)[0:3]
    
    
    for RN in RNG_KNO:
        if normed:
            plt.plot(tt[1:], out_Te[0,1:,RN].T/np.max(out_Te[:,1:,RN].T), c='k', lw=3, alpha=0.2)
            plt.plot(tt[1:], out_St[0,1:,RN].T/np.max(out_St[:,1:,RN]), color=palette[ip])
            #plt.gca().set_prop_cycle(None)
            plt.plot(tt[1:], out_St0[0,1:,RN].T/np.max(out_St0[:,1:,RN]), '--', color=palette[ip])
        else:
            plt.plot(tt[1:], out_Te[0,1:,RN].T, c='k', lw=3, alpha=0.2)
            plt.plot(tt[1:], out_St[0,1:,RN].T, color=palette[ip])
            #plt.gca().set_prop_cycle(None)
            plt.plot(tt[1:], out_St0[0,1:,RN].T, '--', color=palette[ip])
    if normed:
        plt.ylabel('activity (norm.)')
    else:
        plt.ylabel('activity ')
    
    plt.xlabel('time (a.u.)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if not normed:
        ax.set_yticks([0, 1, 2])
        
    plt.savefig('Figs/AshokEI_exampleRec_'+str(Perc)+'.pdf')

    #RNG = [np.argmax(diff[mask_unk])    ,]#[172,]#[105, ]#150]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rng_unk = rn_ar[Perc:]
    #Rng_unk = np.array([116, 67])#rn_ar[Perc:]
    # plt.plot(out_Te[0,:,per7cs[ip]:percs[ip]+12], c='k')
    # plt.plot(out_St[0,:,percs[ip]:percs[ip]+12], '--')
    lss = np.random.permutation(np.arange(percs[ip], Np))[0:2]
    tt = np.arange(len(out_Te[0,:,0]))*0.1/5
    for RN in rng_unk[[-1,-16]]:
        if np.min(np.abs(RN-rng_unk))==0:
            if normed:
                plt.plot(tt[1:], out_Te[0,1:,RN].T/np.max(out_Te[0,1:,RN].T), c='k', lw=3, alpha=0.2)
                plt.plot(tt[1:], out_St[0,1:,RN].T/np.max(out_St[0,1:,RN]), color=palette[ip])
                #plt.gca().set_prop_cycle(None)
                plt.plot(tt[1:], out_St0[0,1:,RN].T/np.max(out_St0[0,1:,RN]), '--', color=palette[ip])
            else:
                plt.plot(tt[1:], out_Te[0,1:,RN].T, c='k', lw=3, alpha=0.2)
                plt.plot(tt[1:], out_St[0,1:,RN].T, color=palette[ip])
                #plt.gca().set_prop_cycle(None)
                plt.plot(tt[1:], out_St0[0,1:,RN].T, '--', color=palette[ip])
                
            # if normed:
            #     plt.plot(tt[1:], out_Te[1,1:,RN].T/np.max(out_Te[1,1:,RN].T), c='k', lw=3, alpha=0.2)
            #     plt.plot(tt[1:], out_St[1,1:,RN].T/np.max(out_St[1,1:,RN]), color=palette[ip])
            #     #plt.gca().set_prop_cycle(None)
            #     plt.plot(tt[1:], out_St0[1,1:,RN].T/np.max(out_St0[1,1:,RN]), '--', color=palette[ip])
            # else:
            #     plt.plot(tt[1:], out_Te[1,1:,RN].T, c='k', lw=3, alpha=0.2)
            #     plt.plot(tt[1:], out_St[1,1:,RN].T, color=palette[ip])
            #     #plt.gca().set_prop_cycle(None)
            #     plt.plot(tt[1:], out_St0[1,1:,RN].T, '--', color=palette[ip])
    
    
    if normed:
        plt.ylabel('activity (norm.)')
    else:
        plt.ylabel('activity ')
    #plt.ylim([0, 1.5])
    plt.xlabel('time (a.u.)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if not normed:
        ax.set_yticks([0, 1, 2])
    plt.savefig('Figs/AshokEI_exampleUnrec_'+str(Perc)+'.pdf')
    plt.show()

#%%
    fig = plt.figure(figsize=[6.5, 3.1])
    ax = fig.add_subplot(131)
    plt.scatter(np.ravel(out_Te[:,:,rn_ar[0:percs[ip]]]), np.ravel(out_St0[:,:,rn_ar[0:percs[ip]]]) , c='k', alpha=0.2, rasterized=True)
    plt.scatter(np.ravel(out_Te[:,:,rn_ar[0:percs[ip]]]), np.ravel(out_St[:,:,rn_ar[0:percs[ip]]]) , c='C0', rasterized=True)

    
    plt.plot([0, 10], [0, 10], c='k')
    plt.xlabel('rec teacher act.')
    plt.ylabel('rec student act.')
    ax = fig.add_subplot(132)
    plt.scatter(np.ravel(out_Te[:,:,rn_ar[percs[ip]:]]), np.ravel(out_St0[:,:,rn_ar[percs[ip]:]]) , c='k', alpha=0.2, rasterized=True)
    plt.scatter(np.ravel(out_Te[:,:,rn_ar[percs[ip]:]]), np.ravel(out_St[:,:,rn_ar[percs[ip]:]]) , c='C0', rasterized=True)

    
    plt.plot([0, 10], [0, 10], c='k')
    plt.xlabel('unr. teacher act.')
    plt.ylabel('unr. student act.')
    
    ax = fig.add_subplot(133)
    plt.plot(losses_kno_, c='k', label='rec')
    plt.plot(losses_unk_, label='unrec')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('iters.')
    #plt.plot([0, 8000],[lu, lu], c='C0' )
    #plt.plot([0, 8000],[lk, lk], c='k' )
    plt.savefig('Figs/AshokEI_example3_rec_'+str(Perc)+'.pdf', dpi=300)
    plt.show()

    # output_train2st = np.zeros((trials, Nt, NT))
    
    # g_St = gS#[:,np.newaxis]).type(dtype)
    # b_St = bS#[:,np.newaxis]).type(dtype)
    # g_St[Np:] = gf[Np:]
    # b_St[Np:] = bf[Np:]

        
    
    # for tr in range(trials):
    #     input_train[tr,:,:] = s[:,tr,:]
    #     xB = np.zeros((len(taus), NT))
    #     xB[0,:] = x0
        
    #     output_train2st[tr,0,:]= relu(xB[0,:])
        
    #     for it, ta in enumerate(taus[:-1]):
            
    #         xB[it+1,:]=xB[it,:]+(dta/tMs)*(-xB[it,:]+np.diag(g_St).dot(J.T.dot(relu(xB[it,:])))   + b_St + s[it,tr,:].dot(wI))
    #         output_train2st[tr,it+1,:] = relu(xB[it+1,:])

    # Am = np.load('mnorder.npz')
    # mnorder = Am['arr_0']

    # or_neur = np.argsort(mnorder[0][0:Nm//2])
    # time = 1.2*taus/dta-0.2#*len(taus)
    # # fig = plt.figure(figsize=[3.5, 3.5])
    # # ax = fig.add_subplot(211)
    # # VM =2.
    # # plt.pcolor(time, np.arange(Nm//2), np.abs(output_train2st[0,:,Np+or_neur]-output_train2[0,:,Np+or_neur]),
    # #            shading='auto', vmin=-0., vmax=VM,  cmap='Reds')
    # # plt.pcolor(time, np.arange(Nm//2)+Nm//2, np.abs(output_train2st[0,:,Np+Nm//2+or_neur]-output_train2[0,:,Np+Nm//2+or_neur]),
    # #            shading='auto', vmin=-0., vmax=VM,  cmap='Reds')
    # # plt.gca().invert_yaxis()
    # # ax = fig.add_subplot(212)
    # # plt.pcolor(time, np.arange(Nm//2), np.abs(output_train2st[1,:,Np+or_neur]-output_train2[1,:,Np+or_neur]),
    # #            shading='auto', vmin=-0., vmax=VM, cmap='Reds')
    # # plt.pcolor(time, np.arange(Nm//2)+Nm//2, np.abs(output_train2st[1,:,Np+Nm//2+or_neur]-output_train2[1,:,Np+Nm//2+or_neur]),
    # #            shading='auto', vmin=-0., vmax=VM, cmap='Reds')
    # # #plt.ylabel('MN index')
    # # plt.gca().invert_yaxis()
    # # plt.xlabel('normalized time')
    # # plt.savefig('Figs/fig_neuraldata_studentMN_rec_'+str(Perc)+'.pdf', transparent=True)
    # # plt.show()
    
    #%%

    # plt.plot(output_train2[0,:,210])
    # plt.plot(output_train2st[0,:,210])
    # #%%
    # plt.plot(output_train2st[1,:,0:Np]-output_train2[0,:,0:Np])
#%%
fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(131)
plt.imshow(out_Te[0,:,percs[ip]:], aspect='auto')
plt.title('teacher')
ax = fig.add_subplot(132)
plt.imshow(out_St[0,:,percs[ip]:], aspect='auto')
plt.title('student')
ax = fig.add_subplot(133)
plt.imshow(out_St0[0,:,percs[ip]:], aspect='auto')
plt.title('student0')
plt.show()
#, np.ravel(out_St0[:,:,percs[ip]:]) , c='k', alpha=0.2, rasterized=True)
#plt.scatter(np.ravel(out_Te[:,:,percs[ip]:]), np.ravel(out_St[:,:,percs[ip]:]) , c='C0', rasterized=True)
#%%
plt.scatter(g_Teacher, g_St)
plt.scatter(g_Teacher, g_St0, c='k')
plt.show()
#%%
Am = np.load(stri+'mnorder.npz')
mnorder = Am['arr_0']

or_neur = np.argsort(mnorder[0][0:Nm//2])
time = 1.2*taus/dta-0.2#*len(taus)
fig = plt.figure(figsize=[3.5, 3.5])
ax = fig.add_subplot(211)
plt.pcolor(time, np.arange(Nm//2), output_train2[0,:,Np+or_neur], shading='auto', vmin=0., vmax=1.5, cmap='Greys')
plt.pcolor(time, np.arange(Nm//2)+Nm//2, output_train2[0,:,Np+Nm//2+or_neur], shading='auto', vmin=0., vmax=1.5, cmap='Greys')
ax = fig.add_subplot(212)
plt.pcolor(time, np.arange(Nm//2), output_train2[1,:,Np+or_neur], shading='auto', vmin=0., vmax=1.5, cmap='Greys')
plt.pcolor(time, np.arange(Nm//2)+Nm//2, output_train2[1,:,Np+Nm//2+or_neur], shading='auto', vmin=0., vmax=1.5, cmap='Greys')
#plt.ylabel('MN index')
plt.gca().invert_yaxis()
plt.xlabel('normalized time')
plt.savefig('Figs/fig_neuraldata_teacherMN.pdf', transparent=True)

#%%
#Jpm_ = Jpm*Jpm_mask
or_neur = np.argsort(np.argmax(out_Te[0,:,0:Np],0))
time = 1.2*taus/dta-0.2#*len(taus)
fig = plt.figure(figsize=[3.5, 3.5])
ax = fig.add_subplot(211)
plt.pcolor(time, np.arange(Np), out_Te[0,:,or_neur], shading='auto', vmin=0.1, vmax=4, cmap='Greys')

ax = fig.add_subplot(212)
plt.pcolor(time, np.arange(Np), out_Te[1,:,or_neur], shading='auto', vmin=0, vmax=4, cmap='Greys')
#plt.ylabel('PMN index')
plt.xlabel('normalized time')
plt.savefig('Figs/fig_neuraldata_teacherPMN.pdf', transparent=True)
plt.show()

# #%%
# #Calculate participation ratio
# X = np.hstack((out_Te[0,:,or_neur], out_Te[1,:,or_neur]))
# #X = X-np.mean(X,axis=1, keepdims=True)

# C = np.dot(X, X.T)
# uC = np.linalg.eigvals(C)
# PR = np.sum(uC)**2/np.sum(uC**2)
# print('Participation ratio:', PR)

# #%%
# plt.figure(figsize=[5,3])
# plt.imshow(X, cmap='bwr')
# plt.colorbar()
# plt.xlabel('timepoints')
# plt.ylabel('neurons')
# plt.title('PR='+str(PR)[0:4])
# plt.savefig('pr_larval.png')
# #%%
# np.savez('pr_x_larval.npz', X)