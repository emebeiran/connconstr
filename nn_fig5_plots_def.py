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
    return (1/beta)*np.log(1+np.exp(beta*x))

def random_group(x, scale=1.):
    groups = [0, 92, 134]#[0, 46, 50, 70, 92, 134]
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
ori0_ = np.linspace(-np.pi*0.5, np.pi*0.5, 8)[2]
AA = np.load('Data/Figure5/params_netSimpleRing2_final.npz')

JJ = AA['arr_0']
gf = AA['arr_1']
bf = AA['arr_2']
x0 = AA['arr_3']
wI = AA['arr_4']
wOut = AA['arr_5']
alpha = AA['arr_6']
si = np.exp(AA['arr_7'])

input_size = np.shape(wI)[0]
hidden_size = np.shape(wOut)[0]
output_size = np.shape(wOut)[1]
wI2 = np.diag(si[:,0]).dot(wI)
Beta = 5.
N = hidden_size
wout = np.eye(hidden_size)

#%%
sig_b = np.std(bf)
sig_g = np.std(gf)

    
#%%
seed = 21
sd = 0.001
n_ep = 210
np.random.seed(seed)
bt = sig_b*np.random.randn(hidden_size)+np.mean(bf)


dtype = torch.FloatTensor  
wi_init = torch.from_numpy(wI).type(dtype)
wo_init = torch.from_numpy(wout[:,np.newaxis]).type(dtype)

wrec_init = torch.from_numpy(JJ).type(dtype)
b_init_True= torch.from_numpy(bf[:,np.newaxis]).type(dtype)

b2_mask = np.arange(len(bf))
b_init_Ran= torch.from_numpy(bf[b2_mask,np.newaxis]).type(dtype)
b_init= torch.from_numpy(bf[:,np.newaxis]).type(dtype)



H0= torch.from_numpy(x0).type(dtype)
#%%
trials = 50
Trials= trials
dta = 0.1

TT = 16.
TStay = 8.

Wbump = 15.
y,m,inps, ts = fs.generate_targets(Trials, wbump=Wbump, T=TT, stay=True, tStay = TStay, ori0=ori0_)
Ntim = len(ts)
taus = ts
Nt = len(taus)
alpha = dta


output_train = np.zeros((trials, Nt, output_size))
output_train2 = np.zeros((trials, Nt, hidden_size))

for tr in range(trials):
    #print(tr)
    
    xB = np.zeros((len(taus), hidden_size))
    rB = np.zeros((len(taus)))
    
    xB[0,:] = x0
    output_train[tr,0,:] = wOut.T.dot(gf*relu(x0+bf))
    
    output_train2[tr,0,:]= gf*relu(xB[0,:]+bf)
    
    for it, ta in enumerate(taus[:-1]):
                                 
        xB[it+1,:]=xB[it,:]+(dta)*(-xB[it,:]+JJ.dot(gf*relu(xB[it,:]+ bf)) + inps[tr,it,:].dot(wI2))
        output_train[tr,it+1,:] = wOut.T.dot(relu(x0+bf))
        output_train2[tr,it+1,:] = gf*relu(xB[it+1,:]+bf)
        

    
#%%

Max = 5.
Min = 0.2
N = np.shape(JJ)[0]
input_size = 46+2
bt = 0.1*np.random.randn(N)
gt = 0.1*np.random.randn(N)-0.2

taus0 = np.arctanh((2-(Max+Min))/(Max-Min))*np.ones(N)
output_size = 3+46
wout = np.zeros((N,output_size))

winp = np.zeros((input_size,N))
sinp = np.zeros((input_size,1))

dtype = torch.FloatTensor 
Wr_ini =JJ
Wr_ini_log = np.copy(JJ)
Wr_ini_log[np.abs(Wr_ini_log)>0] = np.log(np.abs(Wr_ini_log[np.abs(Wr_ini_log)>0] ))

wrec_init = torch.from_numpy(Wr_ini_log).type(dtype)
mwrec_init = torch.from_numpy(np.sign(Wr_ini)).type(dtype)

b_init = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
g_init = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
taus_init = torch.from_numpy(taus0[:,np.newaxis]).type(dtype)


wi_init = torch.from_numpy(winp).type(dtype)
si_init = torch.from_numpy(sinp).type(dtype)
wo_init = torch.from_numpy(wout).type(dtype)

dta = ts[1]-ts[0]
Tau = 1.

alpha = dta/Tau
hidden_size=N


h0_init = torch.from_numpy(x0).type(dtype)
input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(y)

Beta = 5.
lin_var = False

Net = fs.RNN_fly(input_size, hidden_size, output_size, wi_init, si_init, wo_init, wrec_init, mwrec_init, b_init, g_init, taus_init,
          h0_init=h0_init, train_b = False, train_g=False, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
          alpha=alpha, linear=lin_var, train_h0=False, beta=Beta, train_taus=False)
    
#inps[:,:,0:46] = 0.
input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(y)
mask_train = torch.from_numpy(m).type(dtype)
trainstr=''
Ori0_ = np.round(ori0_*10)/10
Net.load_state_dict(torch.load('Data/Figure5/netPopVec_Wrec_SimpleRing.pt', map_location='cpu'))
out2 = Net.forward(input_Train)
out = out2.detach().numpy()

#%%
percs = np.array((1, 2, 4,  6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, ))

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
    aa = np.load(stri+'data_fig5_def.npz')
    losses  = aa['arr_0']
    losses_unk = aa['arr_1']
    losses_kno = aa['arr_2']
except:
    print('loading data')
    for ilr, lr in enumerate(lrs):
        for ic in range(ics):
            for ip in range(len(used_percs)):
    
                
                try:
                    AA = np.load(stri+'taskSimpleRingFMix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
                    # if ic==3 and ip==0:
                    #     AA = np.load(stri+'taskSimpleRingFMix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic-1)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
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
    np.savez(stri+'data_fig5_def.npz', losses, losses_unk, losses_kno)
#%%
losses00 = np.zeros(ics)
for ic in range(ics):
    AA = np.load(stri+'taskSimpleRing2Mix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_00_unpacked.npz')#, losses[::10],  loss_kno, loss_unk, loss_kno_r2, loss_unk_r2)
    if r2:
        losses00[ic] = AA['arr_3'][0]
    else:
        losses00[ic] = AA['arr_1'][0]
    
#%%
losses[losses==0] = np.nan
#losses_read[losses_read==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan


#%%

factor =5
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

AA = np.load('Data/Figure5/loss_SimpleRing2_baselineF.npz')#, loss_bs, loss_bs_r2)
loss_bs = AA['arr_0']
loss_bs_r2 = AA['arr_1']
loss_bs_r2_NoJ = AA['arr_3']


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
    print(' ')
    #ax.plot([percs[0], percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
else:
    ax.plot([percs[0], percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)


plt.savefig('Figs/AshokSimpleRing'+str_r2+'_Mix_unknownLoss_summ.pdf')
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
plt.savefig('Figs/AshokSimpleRingF'+str_r2+'_Mix_knownLoss_summ.pdf')
# # plt.savefig('Figs/SGD_Adam_trial_percs_unknownLoss_final.pdf')

#%%
Lo = losses_unk[:,:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
m_all = []
for ip, pi in enumerate(percs):
    xx = Lo[-1,ip,:]
    mm = np.nanmean(Lo[:,ip,:],-1)#/np.mean(Lo[0,ip,:], -1)
    ss = np.nanstd(Lo[:,ip,:], -1)/np.sqrt(len(percs))#np.std(Lo[:,ip,:]/Lo[0,ip,:], -1)/np.sqrt(len(percs))
    plt.scatter(pi, mm[-2], s=40, color=palette[ip], edgecolor='k', zorder=4)
    plt.plot([pi, pi], [mm[-2]-ss[-1], mm[-2]+ss[-1]], color=palette[ip])
    plt.scatter(pi*np.ones(len(xx)), xx, s=8, color=palette[ip], edgecolor=palette[ip], zorder=4 )
    m_all.append(mm[-2])
plt.plot(percs, m_all, c='k', lw=.5)

mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
plt.scatter(0.5*np.ones(len(losses00)), losses00, s=8, color='k', edgecolor='k', zorder=4 )


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

# if r2:
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs_r2), np.mean(loss_bs_r2)], lw=4, c='k', alpha=0.3)
# else:
#     ax.plot([0.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)
# plt.yscale('log')


plt.savefig('Figs/AshokSimpleRingF'+str_r2+'_Mix_unknownLoss_summ2.pdf')

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
plt.savefig('Figs/AshokSimpleRingF'+str_r2+'_Mix_knownLoss_summ2.pdf')
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
plt.savefig('Figs/AshokSimpleRingF'+str_r2+'_Mix_unknownLoss_summ3.pdf')
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


    if ip==0:        
        mm = np.nanmean(Lo[0,ip,:])#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
        ss = np.nanstd(Lo[0,ip,:])/np.sqrt(ics)
        plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
        plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')
        
        
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


# mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
# ss = np.nanstd(losses00)/np.sqrt(ics)
# plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
# plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')

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
plt.savefig('Figs/AshokSimpleRingF'+str_r2+'_Mix_unknownLoss_summ4.pdf')

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


    if ip==0:        
        mm = np.nanmean(Lo[0,ip,:])#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
        ss = np.nanstd(Lo[0,ip,:])/np.sqrt(ics)
        plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
        plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')
        
        
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
#     ax.plot([.5, percs[-1]], [np.mean(loss_bs_r2_NoJ), np.mean(loss_bs_r2_NoJ)], '--', lw=4, c='k', alpha=0.3)
    
# else:
#     ax.plot([.5, percs[-1]], [np.mean(loss_bs), np.mean(loss_bs)], lw=4, c='k', alpha=0.3)


# mm = np.nanmean(losses00)#/np.mean(Lo[0,ip,:], -1)ss = np.nanstd(losses00)
# ss = np.nanstd(losses00)/np.sqrt(ics)
# plt.scatter(0.5, mm, s=40, color='k', edgecolor='k', zorder=4)
# plt.plot([0.5, 0.5],[mm-ss, mm+ss], color='k')
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
plt.savefig('Figs/AshokSimpleRingF'+str_r2+'_Mix_unknownLoss_summ4_NoJ.pdf')


# #%%
# #Calculate participation ratio
# shap = np.shape(output_train2)
# X = np.zeros((shap[0]*shap[1], shap[2]))
# for i in range(shap[2]):
#     X[:,i]= np.ravel(output_train2[:,:,i])
# X = X.T
# X = X-np.mean(X,axis=1, keepdims=True)

# C = np.dot(X, X.T)
# uC = np.linalg.eigvals(C)
# PR = np.sum(uC)**2/np.sum(uC**2)
# print('Participation ratio:', PR)

# # #%%
# # plt.figure(figsize=[5,3])
# # plt.imshow(X, cmap='bwr')
# # plt.colorbar()
# # plt.xlabel('timepoints')
# # plt.ylabel('neurons')
# # plt.title('PR='+str(PR)[0:4])
# # plt.savefig('pr_larval.png')
# #%%
# np.savez('pr_x_drosophila.npz', X)
#%%
def get_angles(out):
    out = out[:,:,0:2]
    out = out/np.sqrt(np.sum(out**2,-1,keepdims=True))
    angs = np.arctan2(out[:,:,1],out[:,:,0])
    return(angs)
#y,m,inps, ts = fs.generate_targets(Trials, wbump=Wbump, T=TT, stay=True, tStay = TStay, ori0=ori0_)

# #%%
# y3,m3,inps3, ts3 = fs.generate_targetssimple(Trials*4)

# ip=0
# ic  =5
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
#     plt.plot(outt[ITR,-1,3:49].T, c='k', label='teacher')
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
#     plt.imshow(outt[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('teacher', fontsize=14)
#     plt.savefig('simplering2_netAngle_exampleBump3_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
    

# #%%

# ip=10
# ic  =5
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
# plt.plot([-np.pi, np.pi], [-np.pi, np.pi], lw=0.5, c='k')
# plt.xlabel('teacher angle')
# plt.ylabel('student angle')
# plt.yticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_yticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.xticks([-np.pi, -np.pi*0.5, 0., np.pi*0.5, np.pi])
# ax.set_xticklabels([r"$ \pi$", r"$ \pi/2$", "0", r"$\pi /2$", r"$\pi$"])
# plt.savefig('Figs/simplering2_netAngle_exampleBump_trial_'+str(ic)+'_Nrec_'+str(percs[ip])+'.pdf', dpi=400)
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
# plt.savefig('Figs/simplering2_netAngle_exampleBump_clean_Nrec_'+str(percs[ip])+'.pdf', dpi=400)
# plt.show()


# #%%
# ITRs = [3, 4, ]#5, 10, 15]
# for ITR in ITRs:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(out_task[ITR,-1,3:49].T, c=palette[ip], label='after training')
#     plt.plot(out_task0[ITR,-1,3:49].T, '--', c=palette[ip], lw=0.5, label='before training')
#     plt.plot(outt[ITR,-1,3:49].T, c='k', label='teacher')
#     plt.ylabel('activity (end of trial)')
#     plt.xlabel('EPG neuron index')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_ylim([0, 12])
#     plt.legend()
#     plt.savefig('Figs/simplering2_netAngle_exampleBump2_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
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
#     plt.imshow(outt[ITR,:,3:49].T, vmax= VM, vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('teacher', fontsize=14)
#     plt.savefig('Figs/simplering2_netAngle_exampleBump3_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
#     fig = plt.figure(figsize=[10,4])
    
#     ax = fig.add_subplot(131)
#     plt.imshow(out_task[ITR,:,3:49].T,  vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('student after tr.', fontsize=14)
#     ax = fig.add_subplot(132)
#     plt.imshow(out_task0[ITR,:,3:49].T,  vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('student before tr.', fontsize=14)
#     ax = fig.add_subplot(133)
#     plt.imshow(outt[ITR,:,3:49].T,  vmin=Vmi, cmap='Greys')
#     plt.grid(None)
#     ax.set_title('teacher', fontsize=14)
#     plt.savefig('Figs/simplering2_netAngle_exampleBump3_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
#     plt.show()
    
#     #%%
#     ITR = 2
#     IN = 25
#     plt.plot(out_task[ITR,:,IN].T, )
#     plt.plot(out_task0[ITR,:,IN].T, c='k')
#     plt.plot(outt[ITR,:,IN].T, lw=3, alpha=0.5,c='k')
# #%%
#     ITR = 6
#     IN = 5
#     plt.plot(out_task[ITR,:,IN].T, )
#     plt.plot(out_task0[ITR,:,IN].T, c='k')
#     plt.plot(outt[ITR,:,IN].T, lw=3, alpha=0.5,c='k')

#     #%%
#     ITR = 2
#     plt.imshow(out_act0[ITR,:,0:46], cmap='Greys', vmin=0, vmax=5)
#     plt.show()
#     plt.imshow(out_act[ITR,:,0:46], cmap='Greys', vmin=0, vmax=5)
#     plt.show()
#     plt.imshow(out_teach_act[ITR,:,0:46], cmap='Greys', vmin=0, vmax=5)
#     plt.show()
#     #%%
#     ITR = 2
#     plt.imshow(out_task0[ITR,:,0:46], cmap='Greys', vmin=0, vmax=5)
#     plt.show()
#     plt.imshow(out_task[ITR,:,0:46], cmap='Greys', vmin=0, vmax=5)
#     plt.show()
#     plt.imshow(outt[ITR,:,0:46], cmap='Greys', vmin=0, vmax=5)
#     plt.show()

    #%%
    
    
    # plt.ylabel('activity (end of trial)')
    # plt.xlabel('EPG neuron index')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.set_ylim([0, 12])
    # plt.legend()
    # plt.savefig('Figs/simplering2_netAngle_exampleBump2_clean_Nrec_'+str(percs[ip])+'_ITR_'+str(ITR)+'.pdf', dpi=200)
    # plt.show()
