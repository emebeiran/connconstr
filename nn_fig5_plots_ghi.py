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

import sys
import fun_lib3 as fs


def relu(x, beta=5.):
    yy = np.exp(beta*x)
    sol= np.copy(x)    
    sol[yy<100] = (1/beta)*np.log(1+yy[yy<100])
    return sol

def random_group(x, scale=1.):
    groups = [0,92,134]#[0, 46, 50, 70, 92, 134]
    x_rand = np.copy(x)
    
    for ig, gi in enumerate(groups):
        if ig==len(groups)-1:
            real = x[gi:]
            idx = np.arange(gi, len(x))
        else:
            real = x[gi:groups[ig+1]]
            
            idx = np.arange(gi, groups[ig+1])
        x_rand[idx] = scale*np.random.permutation(real)
    return(x_rand)


#%%    
fs.set_plot()

#%%
AA = np.load('Data/Figure5/zebrafish.npz')
W_ = AA['arr_0']
I_ = AA['arr_1']
v_in_ = AA['arr_2']
dt_ = AA['arr_3']


N_ = np.shape(W_)[0]
lent_ = len(I_)

r_ = np.zeros((N_,lent_))
for it in range(lent_-1):
    r_[:,it+1] = r_[:,it]+dt_*(-r_[:,it]+W_.dot(r_[:,it])+I_[it]*v_in_)

#%%
plt.imshow(W_, vmin=-0.01, vmax=0.01, cmap='bwr')
plt.show()

plt.plot(np.sum(np.abs(W_),0))
plt.plot(np.sum(np.abs(W_),1))
plt.show()

#%%
maskn = np.sum(np.abs(W_),0)>0#(np.sum(np.abs(W_),0)+np.sum(np.abs(W_),1))>0
N = np.sum(maskn)
def old2new(I):
    i_old = np.arange(N_)
    i_olds = i_old[maskn]
    aa = np.where(i_olds==I)
    return(aa[0][0])
#272, 273, 281
#%%
W = W_[maskn,:]
W = W[:, maskn]
v_in = v_in_[maskn]



#%%
dt = 0.1
T = dt_*len(I_)

time = np.arange(0,T, dt)
time_ = np.arange(0,T, dt_)

I = np.zeros_like(time)
t0s = [1, 8, 15]
for t0 in t0s:
    ai = np.argmin(np.abs(time-t0))
    I[ai] = 10.5/dt
#%%
N = np.shape(W)[0]
lent = len(I)

r = np.zeros((N,lent))
for it in range(lent-1):
    r[:,it+1] = r[:,it]+dt*(-r[:,it]+W.dot(r[:,it])+I[it]*v_in)

#%%

fig = plt.figure()
ax = fig.add_subplot(111)


neurs = [272, 273, 281]
for iN in neurs:

    plt.plot(time_, r_[iN,:], c=0.1*np.ones(3), lw=1.5)
    iN2 = old2new(iN) 
    #plt.plot(time, r[iN2,:], '--', c='k')


ax.set_ylabel('activity')
ax.set_xlabel(r'time ($\tau$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
#x.set_xscale('log')

plt.savefig('AshokFish2_act.pdf')
plt.show()
#np.savez('zebrafish2.npz', W, I, g, v_in, dt, maskn, N_)

#%%

AA = np.load('Data/Figure5/zebrafish2.npz')

JJ = AA['arr_0']
I = AA['arr_1']
gf = AA['arr_2']
#bf = AA['arr_2']
x0 = np.zeros_like(gf)#AA['arr_3']
wI = AA['arr_3']
maskn = AA['arr_5']
alpha = AA['arr_4']


input_size = 1
hidden_size = np.shape(JJ)[0]
output_size = np.shape(JJ)[0]
wI2 = wI[np.newaxis,:]
Beta = 5.
N = hidden_size
wout = np.eye(hidden_size)[:,0:output_size]

#%%
# sig_b = np.std(bf)
sig_g = np.std(gf)

#%%
mask2 = np.sum(JJ,0)
ia = np.argsort(mask2)
ia=ia[::-1]
JJ2 = np.copy(JJ)
JJ2 = JJ2[ia,:]
JJ2 = JJ2[:,ia]
fig = plt.figure()#figsize=[1.5*3.5, 1.5*2.7])
plt.imshow(JJ2, vmin=-0.05, vmax=0.05, cmap = 'bwr_r')
plt.grid('none')
plt.colorbar()
plt.savefig('AshokSimpleRing_connBar.pdf')
plt.show()


fig = plt.figure()#figsize=[1.5*3.5, 1.5*2.7])
plt.imshow(JJ2, vmin=-0.05, vmax=0.05, cmap = 'bwr_r')
plt.grid('none')
plt.xlabel('presynaptic')
plt.ylabel('postsynaptic')

#plt.colorbar()
plt.savefig('Figs/AshokSimpleRing_conn.pdf')
plt.show()


def old2new(I):
    i_old = np.arange(N_)
    i_olds = i_old[maskn]
    aa = np.where(i_olds==I)
    return(aa[0][0])

#%%
neurs = [272, 273, 281]
for iN in neurs:
    plt.plot(time_, r_[iN,:])
    iN2 = old2new(iN) 
    #plt.plot(time, r[iN2,:], '--', c='k')
plt.show()

#%%
seed = 21
sd = 0.001
n_ep = 210
np.random.seed(seed)


dtype = torch.FloatTensor  
wi_init = torch.from_numpy(wI).type(dtype)
wo_init = torch.from_numpy(wout[:,np.newaxis]).type(dtype)

wrec_init = torch.from_numpy(JJ).type(dtype)
b_init_True= torch.from_numpy(0*gf[:,np.newaxis]).type(dtype)

b_init_Ran= torch.from_numpy(0*gf[:,np.newaxis]).type(dtype)
b_init    = torch.from_numpy(0*gf[:,np.newaxis]).type(dtype)



H0= torch.from_numpy(x0).type(dtype)
#%%
trials = 1
Trials= trials
dta = alpha

TT = len(I)*dta
Nt = len(I)

taus = np.arange(0, TT, dta)
output_train = np.zeros((trials, Nt, output_size))
output_train2 = np.zeros((trials, Nt, hidden_size))
inps = np.zeros((trials, Nt, input_size))
inps[0,:,0]=I

for tr in range(trials):
    #print(tr)
    
    xB = np.zeros((len(taus), hidden_size))
    
    xB[0,:] = x0
    output_train[tr,0,:] = wout.T.dot(x0)
    
    output_train2[tr,0,:]= x0
    
    for it, ta in enumerate(taus[:-1]):
                                 
        xB[it+1,:]=xB[it,:]+(dta)*(-xB[it,:]+JJ.dot(gf*xB[it,:]) + inps[tr,it,:].dot(wI2))
        output_train[tr,it+1,:] = wout.T.dot(xB[it+1,:])
        output_train2[tr,it+1,:] = wout.T.dot(xB[it+1,:])
        
output_trainNoJ = np.zeros((trials, Nt, output_size))

inps = np.zeros((trials, Nt, input_size))
inps[0,:,0]=I

for tr in range(trials):
    #print(tr)
    
    xB = np.zeros((len(taus), hidden_size))
    
    xB[0,:] = x0
    output_trainNoJ[tr,0,:] = wout.T.dot(x0)
    
    #output_train2[tr,0,:]= x0
    
    for it, ta in enumerate(taus[:-1]):
                                 
        xB[it+1,:]=xB[it,:]+(dta)*(-xB[it,:]+0*JJ.dot(gf*xB[it,:]) + inps[tr,it,:].dot(wI2))
        output_trainNoJ[tr,it+1,:] = wout.T.dot(xB[it+1,:])
        #output_train2[tr,it+1,:] = wout.T.dot(xB[it+1,:])
        
#%%
TR = 0
plt.imshow(output_train[TR,:,:].T, vmin = 0, vmax=4.)
plt.show()

#%%
#%%
# #Calculate participation ratio

# X = output_train[0,:,:]
# X = X.T
# X = X-np.mean(X,axis=1, keepdims=True)

# C = np.dot(X, X.T)
# uC = np.linalg.eigvalsh(C)
# PR = np.sum(uC)**2/np.sum(uC**2)
# print('Participation ratio:', PR)

# # # #%%
# # # plt.figure(figsize=[5,3])
# # # plt.imshow(X, cmap='bwr')
# # # plt.colorbar()
# # # plt.xlabel('timepoints')
# # # plt.ylabel('neurons')
# # # plt.title('PR='+str(PR)[0:4])
# # # plt.savefig('pr_larval.png')
# # #%%
# np.savez('pr_x_zebrafish.npz', X)

# # plt.imshow(output_train2[TR,:,0:49].T, vmin = 0, vmax=4.)
# # plt.show()
# # plt.plot(inps[TR,:,-2:])
# # plt.show()

#%%
ts = taus

N = np.shape(JJ)[0]
gt = np.random.permutation(gf)


output_size = N
wout = np.zeros((N,output_size))

winp = np.zeros((input_size,N))
winp[0,:]= wI2
dtype = torch.FloatTensor 
#Wr_ini =JJ
#Wr_ini_log = np.copy(JJ)
#Wr_ini_log[np.abs(Wr_ini_log)>0] = np.log(np.abs(Wr_ini_log[np.abs(Wr_ini_log)>0] ))

wrec_init = torch.from_numpy(JJ).type(dtype)

g_init = torch.from_numpy(gt[:,np.newaxis]).type(dtype)


wi_init = torch.from_numpy(winp).type(dtype)
wo_init = torch.from_numpy(wout).type(dtype)

dta = ts[1]-ts[0]
Tau = 1.

alpha = dta/Tau
hidden_size=N


h0_init = torch.from_numpy(x0).type(dtype)
input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(output_train)
m = np.ones_like(output_train)
Beta = 5.
lin_var = False

Net = fs.RNN_fish(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init,  g_init, 
          h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
          alpha=alpha,  train_h0=False)
    # class RNN_fish(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, 
    #                   g_init, h0_init = None,  train_g=True, train_conn = False, 
    #                  train_wout = False, train_wi = False, train_h0=False, train_taus = True, noise_std=0.005, alpha=0.2,
    #                   linear=True, beta=1.):
            
#inps[:,:,0:46] = 0.
input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(output_train)
mask_train = torch.from_numpy(m).type(dtype)
trainstr=''
out2 = Net.forward(input_Train)
out = out2.detach().numpy()
#%%
plt.plot(out[0,:,0])
plt.plot(output_train[0,:,0])

#%%


iters = 50
loss_bs = np.zeros(iters)#AA['arr_0']
loss_bs_r2 = np.zeros(iters)#AA['arr_1']
loss_bs_r2_noInp = np.zeros(iters)#AA['arr_1']


for itr in range(iters):
    rndrng = np.random.permutation(np.arange(N))
    loss = (output_train[0,:,np.arange(N)]-output_train[0,:,rndrng])**2
    loss = np.mean(np.mean(loss,0),0)
    loss_bs[itr] = np.mean(loss)
    
    otG = output_train[0,:,np.arange(N)]#, output_train_G[1,:,0:]))
    ot2 = output_train[0,:,rndrng]#, output_train2[1,:,0:]))
    CC = np.corrcoef(otG, ot2)
    loss_bs_r2[itr] = 1-np.mean((np.diag(CC[0:N,N:])))                       
    
    otG = output_train[0,:,np.arange(N)]#, output_train_G[1,:,0:]))
    ot2 = output_trainNoJ[0,:,np.arange(N)]#, output_train2[1,:,0:]))
    CC = np.corrcoef(otG, ot2)
    loss_bs_r2_noInp[itr] = 1-np.mean((np.diag(CC[0:N,N:])))                       
    
    # loss_kno[ep]  = np.mean(loss[rn_ar[0:Nkno]])
    # loss_kno_r2[ep]  = 1-np.mean(lossr2[rn_ar[0:Nkno]])
    # if Nkno==NT2:
    #     loss_unk[ep] = np.nan
    #     loss_unk_r2[ep] = np.nan
    # else:
    #     loss_unk[ep] = np.mean(loss[rn_ar[Nkno:N]])
    #     loss_unk_r2[ep] = 1-np.mean(lossr2[Nkno:])
        
    # if ep==0:
    #     output_train_G0 = np.copy(output_train_G)


#%%

# plt.scatter(np.exp(Net.g.detach().numpy()[:,0]), gf)
# plt.show()
# plt.scatter(Net.h0.detach().numpy(), x0)
# plt.show()

# #%%
# Trs = [3, 18]
# iT = 0
# fig = plt.figure()
# ax = fig.add_subplot(121)
# plt.imshow(out[Trs[iT],:,3:].T, aspect='auto', vmax = 0.8*np.max(out[Trs[iT],:,3:].T))
# plt.colorbar()
# ax = fig.add_subplot(122)
# plt.imshow(y[Trs[iT],:,3:].T, aspect='auto', vmax = 0.9*np.max(y[Trs[iT],:,3:].T))
# plt.colorbar()

# fig = plt.figure()
# ax = fig.add_subplot(121)
# plt.imshow(output_train2[Trs[iT],:,0:46].T, aspect='auto', vmax = 0.8*np.max(out[Trs[iT],:,3:].T))
# plt.colorbar()
# ax = fig.add_subplot(122)
# plt.imshow(y[Trs[iT],:,3:].T, aspect='auto', vmax = 0.9*np.max(y[Trs[iT],:,3:].T))
# plt.colorbar()
# #%%
# NN = 23
# IT = 17
# #plt.plot(out[IT,:,3+NN]-output_train2[IT,:,NN], c='k')
# plt.plot(out[IT,:,3+NN], c='k')
# plt.plot(output_train2[IT,:,NN], c='C0')
# plt.show()

# #%%
# plt.scatter(np.ravel(out[:,0,3:49]), np.ravel(output_train2[:,0,0:46]))
# plt.plot([0, 4], [0,4])

# #%%
# plt.hist(np.ravel(out[:,0,3:49])- np.ravel(output_train2[:,0,0:46]))


#%%
percs = np.array((1, 2, 4, 8, 12, 16, 24, 32, 48, 56, 64, 96,  128, 160, 196, 228, 260,))


#%%
lrs = [ 0.001, ] #ne
lr = lrs[0]
save_neurs = 100

count = 0
used_percs = percs
e=False
ics = 20
stri = 'Data/Figure5/'
inf_sort = True
r2 = True
if r2:
    str_r2 = ''
else:
    str_r2 = '_MSE_'
try:
    aa = np.load(stri+'data_fig5_ghi.npz')
    losses = aa['arr_0']
    losses_unk = aa['arr_1'] 
    losses_kno = aa['arr_2']
except:
    print('loading losses')

    for ilr, lr in enumerate(lrs):
        for ic in range(ics):
            for ip in range(len(used_percs)):
    
                
                try:
                    AA = np.load(stri+'taskFish_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
                    # if ic==3 and ip==0:
                    #     AA = np.load(stri+'taskSimpleRing2Mix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic-1)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
                except:
                    print('ic: '+str(ic)+'. ip: '+str(ip))
                    
                losses_ = AA['arr_0']
                if r2:
                    losses_kno_ = AA['arr_3']
                    losses_unk_ = AA['arr_4']
                else:
                    losses_kno_ = AA['arr_1']
                    losses_unk_ = AA['arr_2']
                
                if count==0:
                    losses = np.zeros((len(losses_), len(used_percs), ics, len(lrs)))
                    losses_unk = np.zeros((len(losses_unk_), len(used_percs), (ics), len(lrs)))
                    losses_kno = np.zeros((len(losses_kno_), len(used_percs), (ics), len(lrs)))
                    count +=1
                    
    
                if e:
                    losses[:,ip, ic, ilr] = np.nan
                    losses_unk[:,ip, ic, ilr] = np.nan
                    losses_kno[:,ip, ic, ilr] = np.nan
                    
                else:
                    losses[:,ip, ic, ilr] = losses_
                    losses_unk[:,ip, ic, ilr] = losses_unk_
                    losses_kno[:,ip, ic, ilr] = losses_kno_
                    
            
    np.savez(stri+'data_fig5_ghi.npz', losses, losses_unk, losses_kno)
#%%
losses00 = np.zeros(ics)
# for ic in range(ics):
#     AA = np.load(stri+'taskFish_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_00_unpacked.npz')#, losses[::10],  loss_kno, loss_unk, loss_kno_r2, loss_unk_r2)
#     if r2:
#         losses00[ic] = AA['arr_3'][0]
#     else:
#         losses00[ic] = AA['arr_1'][0]
    
#%%
losses[losses==0] = np.nan
#losses_read[losses_read==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan


#%%

factor =1
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
    xx = np.arange(len(mm))*factor
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('$loss_{known}$')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_yscale('log')
#ax.set_yticks([0.01, 0.1, 1.])
plt.savefig('Figs/AshokSimpleRing'+str_r2+'_Mix_knownLoss.pdf')


#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure(figsize=[1.5*3.5, 1.5*2.7])
ax = fig.add_subplot(212)
for ip, pi in enumerate(percs):
    mm = np.mean(Lo[:,ip,:],-1)/np.mean(Lo[0,ip,:], -1)
    ss = np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    xx = np.arange(len(mm))*factor
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('norm.')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
#ax.set_yticks([0.01, 0.1,])
ax = fig.add_subplot(211)
for ip, pi in enumerate(percs):
    mm = np.mean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.std(Lo[:,ip,:], -1)/np.sqrt(len(percs))
    xx = np.arange(len(mm))*factor
    plt.plot(xx, mm, c=palette[ip])
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('$loss_{unknown}$')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')     
ax.set_yscale('log') 
#ax.set_yticks([0.01, 0.1,])
plt.savefig('Figs/AshokSimpleRing'+str_r2+'_Mix_unknownLoss.pdf')


#%%

# AA = np.load('loss_simplering2_baseline.npz')#, loss_bs, loss_bs_r2)
# loss_bs = AA['arr_0']
# loss_bs_r2 = AA['arr_1']

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
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
#plt.yscale('log')

if r2:
    #ax.set_ylim([-0.02, 0.35])
    print('')
    #ax.plot([percs[0], percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
else:
    ax.plot([percs[0], percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)


plt.savefig('Figs/AshokFish'+str_r2+'_Mix_unknownLoss_summ.pdf')
# # plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')

#%%
Lo = losses_kno[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)
ax.set_ylabel('$ loss_{unknown}(T)$')
ax.set_xlabel('known $N$')
ax.set_ylabel('$loss_{known}$')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
#plt.yscale('log')
# if r2:
#     ax.set_ylim([-0.02, 0.35])
plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_knownLoss_summ.pdf')
# # plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
m0 = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-2], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-2]-ss[-1], mm[-2]+ss[-1]], color=palette[ip])
    plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-2])
    if ip==0:
       m0.append(Lo[0,ip,:]) 
plt.plot(percs, m_all, c='k', lw=.5)

mm = np.nanmean(m0)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
plt.scatter(0.5*np.ones(len(m0[0])), m0, s=8, color='k', edgecolor='k', zorder=4 )


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
ax.set_yscale('log')


plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_unknownLoss_summ2.pdf')

#%%
Lo = losses_kno[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=.5)

mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
plt.scatter(0.5*np.ones(len(losses00)), losses00, s=8, color='k', edgecolor='k', zorder=4 )

# yyll = ax.get_ylim()
# if yyll[0]<1e-3:
#     ax.set_ylim([1e-3, yyll[1]])
if r2:
    ax.plot([0.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)

else:
    ax.plot([0.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)
    plt.yscale('log')
    plt.ylim([1e-3, 5e1])

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
#plt.yscale('log')
plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_knownLoss_summ2.pdf')
#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    #plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)


ax.set_ylabel('$loss_{unknown}$')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')


plt.yscale('log')
if r2:
    #ax.set_ylim([-0.02, 0.35])
    ax.plot([1, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
else:
    ax.plot([1, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)
plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_unknownLoss_summ3.pdf')
# plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    #plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)
ax.set_xticks([0.5, 1, 10, 100])
ax.set_xticklabels(['0', '1', '10', '100'])
# mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
# ss = np.nanstd(losses00)/np.sqrt(ics)
# plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
# plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')
# if r2:
#     #ax.set_ylim([-0.02, 0.35])
#     ax.plot([.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
# else:
#     ax.plot([.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)

mm = np.nanmean(m0)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
ss = np.nanstd(m0)/np.sqrt(ics)
plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
plt.plot([.5, .5], [mm-ss, mm+ss], color='k')
#plt.scatter(0.5*np.ones(len(m0[0])), m0, s=8, color='k', edgecolor='k', zorder=4 )

ax.set_ylabel(r'1-$\left[\rho\right]$ (unrec. activity)')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
#plt.yscale('log')
# if r2:
#     #ax.set_ylim([-0.02, 0.35])
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
# else:
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)
plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_unknownLoss_summ4.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    #plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)
ax.set_xticks([0.5, 1, 10, 100])
ax.set_xticklabels(['0', '1', '10', '100'])
# mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
# ss = np.nanstd(losses00)/np.sqrt(ics)
# plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
# plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')
if r2:
    #ax.set_ylim([-0.02, 0.35])
    ax.plot([.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
    ax.plot([.5, percs[-1]], [np.mean(loss_bs_r2_noInp), np.mean(loss_bs_r2_noInp)], '--', lw=4, c='k', alpha=0.3)
    
else:
    ax.plot([.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)

mm = np.nanmean(m0)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
plt.scatter(0.5*np.ones(len(m0[0])), m0, s=8, color='k', edgecolor='k', zorder=4 )

ax.set_ylabel(r'1-|$\rho$| (unrec. activity)')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
#plt.yscale('log')
# if r2:
#     #ax.set_ylim([-0.02, 0.35])
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
# else:
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)
plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_unknownLoss_summ4_NoJ.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(ics)#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-1], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-1]-ss[-1], mm[-1]+ss[-1]], color=palette[ip])
    #plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-1])
plt.plot(percs, m_all, c='k', lw=0.5)
ax.set_xticks([0.5, 1, 10, 100])
ax.set_xticklabels(['0', '1', '10', '100'])
# mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
# ss = np.nanstd(losses00)/np.sqrt(ics)
# plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
# plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')
if r2:
    #ax.set_ylim([-0.02, 0.35])
    ax.plot([.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
else:
    ax.plot([.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)

mm = np.nanmean(m0)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
plt.scatter(0.5*np.ones(len(m0[0])), m0, s=8, color='k', edgecolor='k', zorder=4 )

ax.set_ylabel('error unrec. activity')
ax.set_xlabel('recorded neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(percs[::2])
ax.set_xscale('log')
plt.yscale('log')
# if r2:
#     #ax.set_ylim([-0.02, 0.35])
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
# else:
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)
plt.savefig('Figs/AshokFish2'+str_r2+'_Mix_unknownLoss_summ4_log.pdf')
# #%%
# def get_angles(out):
#     out = out[:,:,0:2]
#     out = out/np.sqrt(np.sum(out**2,-1,keepdims=True))
#     angs = np.arctan2(out[:,:,1],out[:,:,0])
#     return(angs)
# #y,m,inps, ts = fs.generate_targets(Trials, wbump=Wbump, T=TT, stay=True, tStay = TStay, ori0=ori0_)

# #%%
# y3,m3,inps3, ts3 = fs.generate_targetssimple(Trials*4)

# ip=0
# ic  =2
# input_Train3 = torch.from_numpy(inps3).type(dtype)
# out_Train3 = torch.from_numpy(y3)
# mask_train3 = torch.from_numpy(m3).type(dtype)
  
# stt = "Data/netSimpleRing2Mix_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'.pt'
# stt0 = "Data/netSimpleRing2Mix_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_00.pt'#trainstr+'netPopVec_Wrec_timeconstSimple_long_wbump_'+str(Wbump)+'_ori_'+str(ori0_)+'_final.pt'
# output_size = percs[ip]

# wout = np.zeros((N,output_size))
# wo_init = torch.from_numpy(wout).type(dtype)
# wout0 = np.zeros((N,N))
# wo_init0 = torch.from_numpy(wout0).type(dtype)
# Net3 = fs.RNN_fly(input_size, hidden_size, output_size, wi_init, si_init, wo_init, wrec_init, mwrec_init, b_init, g_init, taus_init,
#           h0_init=h0_init, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)
    
# Net3.load_state_dict(torch.load(stt, map_location='cpu'))
# Net0 = fs.RNN_fly(input_size, hidden_size, N, wi_init, si_init, wo_init0, wrec_init, mwrec_init, b_init, g_init, taus_init,
#           h0_init=h0_init, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)

# Net0.load_state_dict(torch.load(stt0, map_location='cpu'))
# output_size = N
# wout = np.eye(N)
# wo_init = torch.from_numpy(wout).type(dtype)
# NetTeacher = Net

# NetTeacherAct = fs.RNN_fly(input_size, hidden_size, output_size, Net.wi, Net.si, wo_init, Net.wrec, Net.mwrec, Net.b, Net.g, Net.taus,
#           h0_init=Net.h0, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)
# NetStudent = fs.RNN_fly(input_size, hidden_size, 49, Net3.wi, Net3.si, Net.wout, Net3.wrec, Net3.mwrec, Net3.b, Net3.g, Net3.taus,
#           h0_init=Net3.h0, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)


# NetStudentAct = fs.RNN_fly(input_size, hidden_size, N, Net3.wi, Net3.si, wo_init, Net3.wrec, Net3.mwrec, Net3.b, Net3.g, Net3.taus,
#           h0_init=Net3.h0, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)

# NetStudent0 = fs.RNN_fly(input_size, hidden_size, 49, Net0.wi, Net0.si, Net.wout, Net0.wrec, Net0.mwrec, Net0.b, Net0.g, Net0.taus,
#           h0_init=Net0.h0, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)


# NetStudentAct0 = fs.RNN_fly(input_size, hidden_size, N, Net0.wi, Net0.si, wo_init, Net0.wrec, Net0.mwrec, Net0.b, Net0.g, Net0.taus,
#           h0_init=Net0.h0, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
#           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)

    
# out2 = Net3.forward(input_Train3)
# out = out2.detach().numpy()
# outT = Net.forward(input_Train3)
# outt = outT.detach().numpy()

# out_task = NetStudent.forward(input_Train3)
# out_task = out_task.detach().numpy()
# out_task0 = NetStudent0.forward(input_Train3)
# out_task0 = out_task0.detach().numpy()


# out_act = NetStudentAct.forward(input_Train3)
# out_act = out_act.detach().numpy()
# out_act0 = NetStudentAct0.forward(input_Train3)
# out_act0 = out_act0.detach().numpy()

# out_teach_act = NetTeacherAct.forward(input_Train3)
# out_teach_act = out_teach_act.detach().numpy()
# Aout_student = get_angles(out_task)
# Aout_student0 = get_angles(out_task0)

# Atar = get_angles(outt)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(np.ravel(Atar[:,-30:]),np.ravel(Aout_student0[:,-30:]), rasterized=True, c='k', alpha=0.05)
# plt.scatter(np.ravel(Atar[:,-30:]),np.ravel(Aout_student[:,-30:]), c=palette[ip], edgecolor='w', alpha=.1, rasterized=True)
# plt.plot([-np.pi, np.pi], [-np.pi, np.pi], lw=0.2)
# plt.xlabel('teacher angle')
# plt.ylabel('student angle')
# plt.yticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_yticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.xticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_xticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.savefig('simplering2_netAngle_exampleBump_trial_'+str(ic)+'_Nrec_'+str(percs[ip])+'.pdf', dpi=200)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# #plt.scatter(np.ravel(Atar),np.ravel(Aout_student0), rasterized=True, c='k', alpha=0.05)
# plt.scatter(np.ravel(Atar[:,-30:]),np.ravel(Aout_student[:,-30:]),  c=palette[ip], edgecolor='w', linewidth=.1, alpha=.1, rasterized=True)
# plt.plot([-np.pi, np.pi], [-np.pi, np.pi], lw=0.2)
# plt.xlabel('teacher angle')
# plt.ylabel('student angle')
# plt.yticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_yticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.xticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_xticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.savefig('simplering2_netAngle_exampleBump_clean_Nrec_'+str(percs[ip])+'.pdf', dpi=200)
# plt.show()

# #%%

# ITRs = [3, 4, 5, 10, 15]
# for ITR in ITRs:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(out_task[ITR,-1,3:49].T, c=palette[ip], label='after training')
#     plt.plot(out_task0[ITR,-1,3:49].T, '--', c=palette[ip], lw=0.5, label='before training')
#     plt.plot(out_teach_act[ITR,-1,3:49].T, c='k', label='teacher')
#     plt.ylabel('activity (end of trial)')
#     plt.xlabel('EPG neuron index')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_ylim([0, 12])
#     plt.legend()
#     plt.savefig('simplering2_netAngle_exampleBump2_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
    
#     VM = 5
#     Vmi = 0
#     fig = plt.figure(figsize=[10,4])
    
#     ax = fig.add_subplot(131)
#     plt.imshow(out_task[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('student after tr.', fontsize=14)
#     ax = fig.add_subplot(132)
#     plt.imshow(out_task0[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('student before tr.', fontsize=14)
#     ax = fig.add_subplot(133)
#     plt.imshow(out_teach_act[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('teacher', fontsize=14)
#     plt.savefig('simplering2_netAngle_exampleBump3_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
    

#%%


for ip in range(len(percs)):
    if np.mod(ip, 5)==0:
        ic  =0
        print(percs[ip])
        # input_Train3 = torch.from_numpy(inps3).type(dtype)
        # out_Train3 = torch.from_numpy(y3)
        # mask_train3 = torch.from_numpy(m3).type(dtype)
          
        stt = "Data/Figure5/netFish_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'.pt'
        stt0 = "Data/Figure5/netFish_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'00.pt'#trainstr+'netPopVec_Wrec_timeconstSimple_long_wbump_'+str(Wbump)+'_ori_'+str(ori0_)+'_final.pt'
        output_size = percs[ip]
        
        wout = np.zeros((N,output_size))
        wo_init = torch.from_numpy(wout).type(dtype)
        wout0 = np.eye(N)
        wo_init0 = torch.from_numpy(wout0).type(dtype)
        Net3 = fs.RNN_fish(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init,  g_init, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        
        # fs.RNN_fly(input_size, hidden_size, output_size, wi_init, si_init, wo_init, wrec_init, mwrec_init, b_init, g_init, taus_init,
        #           h0_init=h0_init, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
        #           alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)
            
        Net3.load_state_dict(torch.load(stt, map_location='cpu'))
        Net0 = fs.RNN_fish(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init,  g_init, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        Net0.load_state_dict(torch.load(stt0, map_location='cpu'))
        
        wo_init = torch.from_numpy(wout).type(dtype)
        NetTeacher = Net
        NetTeacherAct = fs.RNN_fish(input_size, hidden_size, N, Net.wi, wo_init0, Net.wrec,  Net.g, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        
        NetStudent = fs.RNN_fish(input_size, hidden_size, output_size, Net3.wi, Net3.wout, Net3.wrec,  Net3.g, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        
        NetStudentAct = fs.RNN_fish(input_size, hidden_size, N, Net3.wi, wo_init0, Net3.wrec,  Net3.g, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        
        NetStudent0 = fs.RNN_fish(input_size, hidden_size, output_size, Net3.wi, Net3.wout, Net3.wrec,  Net0.g, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        
        
        NetStudentAct0 = fs.RNN_fish(input_size, hidden_size, N, Net3.wi, wo_init0, Net3.wrec,  Net0.g, 
                  h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
                  alpha=alpha,  train_h0=False)
        
        
            
        out2 = Net3.forward(input_Train)
        out = out2.detach().numpy()
        outT = Net.forward(input_Train)
        outt = outT.detach().numpy()
        
        out_task = NetStudent.forward(input_Train)
        out_task = out_task.detach().numpy()
        out_task0 = NetStudent0.forward(input_Train)
        out_task0 = out_task0.detach().numpy()
        
        
        out_act = NetStudentAct.forward(input_Train)
        out_act = out_act.detach().numpy()
        out_act0 = NetStudentAct0.forward(input_Train)
        out_act0 = out_act0.detach().numpy()
        
        out_teach_act = NetTeacherAct.forward(input_Train)
        out_teach_act = out_teach_act.detach().numpy()
        
        rec_neurs = np.zeros((np.shape(Net3.wout)[1])).astype(np.int16)
        
        for ii in range(len(rec_neurs)):
            rec_neurs[ii]=int(np.argmax(Net3.wout[:,ii]))
    
        unrec_neurs = np.arange(N)
    
        for ii in range(len(rec_neurs)):
            IX = np.where(rec_neurs[ii]==unrec_neurs)[0][0]
            unrec_neurs = np.delete(unrec_neurs, IX)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        
        neurs = [272, 273, 281]
        for iN in neurs:
        
            #plt.plot(time_, out_teach_act[iN,:], c=0.1*np.ones(3), lw=1.5)
            iN2 = old2new(iN) 
            if not np.min(np.abs(rec_neurs-iN2))==0:
            #plt.plot(time, out_teach_act[0,:,iN2], '--', c='k')
            
                Mact = 1#np.mean(out_act[0,:,:])
                Mact0 = 1#np.mean(out_act0[0,:,:])
                Mt = 1#np.mean(output_train[0,:,:])
                
                plt.plot(time, out_act[0,:,iN2]/Mact,  c=palette[ip])
                plt.plot(time, out_act0[0,:,iN2]/Mact0,  '--', c=palette[ip])
                plt.plot(time, output_train[0,:,iN2]/Mt, c='k', lw=3, alpha=0.2)
                
            
        
        ax.set_ylabel('activity')
        ax.set_xlabel(r'time ($\tau$)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_xticks(percs[::2])
        #x.set_xscale('log')
        
        plt.savefig('Figs/AshokFish2_percs_'+str(percs[ip])+'_act.pdf')
        plt.show()
    
        fig = plt.figure()
        ax = fig.add_subplot(121)
    
        ax.scatter( np.ravel(output_train[0,:,rec_neurs]), np.ravel(out_act0[0,:,rec_neurs]), c='k')
        ax.scatter( np.ravel(output_train[0,:,rec_neurs]), np.ravel(out_act[0,:,rec_neurs]))
        plt.plot([0, 10], [0, 10])
        ax = fig.add_subplot(122)
    
        ax.scatter( np.ravel(output_train[0,:,unrec_neurs]), np.ravel(out_act0[0,:,unrec_neurs]), c='k') 
        ax.scatter( np.ravel(output_train[0,:,unrec_neurs]), np.ravel(out_act[0,:,unrec_neurs])) 
        plt.plot([0, 10], [0, 10])
        plt.show()
#%%
# #%%
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(np.ravel(Atar[:,-30:]),np.ravel(Aout_student0[:,-30:]), rasterized=True, c='k', alpha=0.05)
# plt.scatter(np.ravel(Atar[:,-30:]),np.ravel(Aout_student[:,-30:]), c=palette[ip], edgecolor='w', alpha=.1, rasterized=True)
# plt.plot([-np.pi, np.pi], [-np.pi, np.pi], lw=0.5, c='k')
# plt.xlabel('teacher angle')
# plt.ylabel('student angle')
# plt.yticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_yticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.xticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_xticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.savefig('fish_exampleBump_trial_'+str(ic)+'_Nrec_'+str(percs[ip])+'.pdf', dpi=400)
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# #plt.scatter(np.ravel(Atar),np.ravel(Aout_student0), rasterized=True, c='k', alpha=0.05)
# plt.scatter(np.ravel(Atar[:,-30:]),np.ravel(Aout_student[:,-30:]), c=palette[ip], edgecolor='w', linewidth=0.1, alpha=.1, rasterized=True)
# plt.plot([-np.pi, np.pi], [-np.pi, np.pi], lw=0.5, c='k')
# plt.xlabel('teacher angle')
# plt.ylabel('student angle')
# plt.yticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_yticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.xticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_xticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.savefig('fish_exampleBump_clean_Nrec_'+str(percs[ip])+'.pdf', dpi=400)
# plt.show()


# #%%
# ITRs = [3, 4, 5, 10, 15]
# for ITR in ITRs:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(out_task[ITR,-1,3:49].T, c=palette[ip], label='after training')
#     plt.plot(out_task0[ITR,-1,3:49].T, '--', c=palette[ip], lw=0.5, label='before training')
#     plt.plot(out_teach_act[ITR,-1,3:49].T, c='k', label='teacher')
#     plt.ylabel('activity (end of trial)')
#     plt.xlabel('EPG neuron index')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_ylim([0, 12])
#     plt.legend()
#     plt.savefig('simplering2_netAngle_exampleBump2_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
# #%%
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(out_task[ITR,-1,3:49].T, c=palette[ip], label='after training')
#     plt.plot(out_task0[ITR,-1,3:49].T, '--', c=palette[ip], lw=0.5, label='before training')
#     plt.plot(out_teach_act[ITR,-1,3:49].T, c='k', label='teacher')
#     plt.ylabel('activity (end of trial)')
#     plt.xlabel('EPG neuron index')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_ylim([0, 12])
#     plt.legend()
#     plt.savefig('simplering2_netAngle_exampleBump2_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
    
    
#     VM = 5
#     Vmi = 0
#     fig = plt.figure(figsize=[10,4])
    
#     ax = fig.add_subplot(131)
#     plt.imshow(out_task[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('student after tr.', fontsize=14)
#     ax = fig.add_subplot(132)
#     plt.imshow(out_task0[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('student before tr.', fontsize=14)
#     ax = fig.add_subplot(133)
#     plt.imshow(out_teach_act[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('teacher', fontsize=14)
#     plt.savefig('simplering2_netAngle_exampleBump3_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
    
#     # plt.ylabel('activity (end of trial)')
#     # plt.xlabel('EPG neuron index')
#     # ax.spines['top'].set_visible(False)
#     # ax.spines['right'].set_visible(False)
#     # ax.yaxis.set_ticks_position('left')
#     # ax.xaxis.set_ticks_position('bottom')
#     # ax.set_ylim([0, 12])
#     # plt.legend()
#     # plt.savefig('simplering2_netAngle_exampleBump2_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     # plt.show()
