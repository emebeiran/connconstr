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
import matplotlib.patches as patches

#from argparse import ArgumentParser
import sys
import fun_lib as fl
fl.set_plot()
import seaborn as sb
#%%
fig = plt.figure()

ax = fig.add_subplot(111)

betas = np.array((0.25, 0.5, 1., 2., 4))
palette2 = sb.diverging_palette(250, 30, l=65, center="dark", n=len(betas)) #sb.color_palette('icefire', n_colors=len(betas))#(np.linspace(0.2, 0.8, len(betas)))#plt.cm.coolwarm(np.linspace(0.2, 0.8, len(betas)))
x = np.linspace(-6, 6, 500)
yteach = np.log(1+np.exp(x))
ystu2   = (1/4)*np.log(1+np.exp(4*x))
ax.plot(x, ystu2, c=palette2[-1], lw=2, label=r'$\beta=4$')
ystu2   = (1/2)*np.log(1+np.exp(2*x))
ax.plot(x, ystu2, c=palette2[-2], lw=2, label=r'$\beta=2$')

ax.plot(x, yteach, c=palette2[-3], lw=2, label='teacher')
ystu2   = (1/0.5)*np.log(1+np.exp(0.5*x))
ax.plot(x, ystu2,  c=palette2[-4], lw=2, label=r' $\beta=0.5$')

ystu2   = (1/0.25)*np.log(1+np.exp(0.25*x))
ax.plot(x, ystu2,  c=palette2[-5], lw=2, label=r' $\beta=0.25$')



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xlabel(r'input $x$')
ax.set_ylabel(r'output $\phi\left(x\right)$')
#ax.legend(frameon=False, fontsize=12, loc=7)
#ax.set_xlim([-6,10])
plt.savefig('Figs/F6_illusA.pdf')

#%%
fig = plt.figure()

ax = fig.add_subplot(111)

betas = np.array((0.25, 0.5, 1., 2., 4))
palette2 = sb.diverging_palette(250, 30, l=65, center="dark", n=len(betas)) #sb.color_palette('icefire', n_colors=len(betas))#(np.linspace(0.2, 0.8, len(betas)))#plt.cm.coolwarm(np.linspace(0.2, 0.8, len(betas)))
x = np.linspace(-6, 6, 500)
yteach = np.log(1+np.exp(x))
ystu2   = (1/4)*np.log(1+np.exp(4*x))
ax.plot(x, ystu2, c=palette2[-1], lw=2, label=r'$\beta=4$')
ystu2   = (1/2)*np.log(1+np.exp(2*x))
ax.plot(x, ystu2, c=palette2[-2], lw=2, label=r'$\beta=2$')

ax.plot(x, yteach, c=palette2[-3], lw=2, label=r'$\beta=1$ (teacher)')
ystu2   = (1/0.5)*np.log(1+np.exp(0.5*x))
ax.plot(x, ystu2,  c=palette2[-4], lw=2, label=r' $\beta=0.5$')

ystu2   = (1/0.25)*np.log(1+np.exp(0.25*x))
ax.plot(x, ystu2,  c=palette2[-5], lw=2, label=r' $\beta=0.25$')



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xlabel(r'input $x$')
ax.set_ylabel(r'output $\phi\left(x\right)$')
ax.legend(frameon=False, fontsize=12, loc=7)
ax.set_xlim([-16,-10])
plt.savefig('Figs/F6_illusA_2.pdf')

#%%
N = 300
dta = 0.05
taus = np.arange(0, 20, dta)
Nt = len(taus)
Tau = 0.2

trials = 60
stri = 'Data/Figure4/' 

seed =21
np.random.seed(seed)

#generate trials
input_train, output_train, cond_train = fl.create_input(taus, trials)



label = ''#'_random'#''
beta = 0.25 #0.5 or 2

lr = 0.002
#%%
f_net = np.load(stri+'teacherRNN_cycling.npz')
#f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
J = f_net['arr_0']
gt = f_net['arr_1']
bt = f_net['arr_2']
wI = f_net['arr_3']
wout = f_net['arr_4']
x0 = f_net['arr_5']
mwrec_ = f_net['arr_6']
refEI = f_net['arr_7']
refVec = refEI[0,:]

alpha = dta

#%%

# parser = ArgumentParser(description='Train on a task using SGD using only a few neurons as readout.')
# parser.add_argument('--seed', required=True, help='Random seed', type=int)
# parser.add_argument('--ip', required=True, help='number of neurons in readout', type=int)
# #parser.add_argument('--ilr', required=True, help='learning rate', type=int)

# args = vars(parser.parse_args())
# ic = args['seed']
# ip = args['ip']  #ilr = args['ilr']
#print(sys.argv)
allseeds = 10#int(sys.argv[2])
percs = np.array((1, 2, 4, 8,10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 150, 200, 250, 300))
#np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40, 80, 150, 200, 250, 300))#np.array((0.0, 1/600, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1,  0.2, 0.5,  1.0))

#np.array((1, 2, 4, 8,10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 150, 200, 250, 300))



#%%
ics = 10

losses_ = []
eG_ = []

losses = []
eG = []

try:
    aa = np.load(stri+'losses_plots_bc.npz')
    losses = aa['arr_0']
    losses_unk = aa['arr_1']
    losses_kno = aa['arr_2']
    losses_G = aa['arr_3']
    losses_B = aa['arr_4']
    #meG = aa['arr_5']
    #sameG = aa['arr_6']

except:
    save_neurs = 100

    count = 0
    used_percs = percs#[0:16]
    e=False
    for ib, be in enumerate(betas):
        for ic in range(ics):
            for ip in range(len(percs)):
                try:
                    AA = np.load(stri+'task'+label+'_lossesBeta'+str(be)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                    losses_ = AA['arr_0']
                    #losses_read_ = AA['arr_1']
                    losses_unk_ = AA['arr_1']
                    losses_kno_ = AA['arr_2']
                    losses_G_ = AA['arr_3']
                    losses_B_ = AA['arr_4']
                    e_g = AA['arr_5']
                    e_b = AA['arr_6']
                    
                    if len(losses_)>700:
                        a = RaiseError[0]
                except:
                    print('ic '+str(ic))
                    print('nRec '+str(used_percs[ip]))
                    print('error --')
                    e=True
        
                
                if count==0:
                    losses = np.zeros((len(losses_), len(used_percs), ics, len(betas) ))
                    losses_unk = np.zeros((len(losses_unk_), len(used_percs), (ics), len(betas) ))
                    losses_kno = np.zeros((len(losses_kno_), len(used_percs), (ics), len(betas) ))
                    losses_G = np.zeros((len(losses_kno_), len(used_percs), (ics), len(betas) ))
                    losses_B = np.zeros((len(losses_kno_), len(used_percs), (ics), len(betas) ))
                    
                    meG = np.zeros((np.shape(e_g)[1], len(used_percs), (ics), len(betas) ))
                    #sameG = np.zeros((save_neurs, np.shape(e_g)[1], len(used_percs), (ics), len(betas) ))
                    
                    #i_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                    #f_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                if e:
                    losses[:,ip, ic, ib] = np.nan
                    losses_unk[:,ip, ic, ib] = np.nan
                    losses_kno[:,ip, ic, ib] = np.nan
                    losses_G[:,ip, ic, ib] = np.nan
                    losses_B[:,ip, ic, ib] = np.nan
                    meG[:,ip, ic, ib] = np.nan
                    #i_eG[:,:,ip, ic, ilr] = np.nan
                    #f_eG[:,:,ip, ic, ilr] = np.nan
                    #sameG[:,ip, ic, ilr] = np.nan
                    
                else:
                    losses[:,ip, ic, ib] = losses_
                    losses_unk[:,ip, ic, ib] = losses_unk_
                    losses_kno[:,ip, ic, ib] = losses_kno_
                    losses_G[:,ip, ic, ib] = losses_G_
                    losses_B[:,ip, ic, ib] = losses_B_
                    meG[:,ip, ic, ib] = np.sqrt(np.mean(e_g**2,axis=(0)))
                    #i_eG[:,:,ip, ic, ilr] = e_g[:,:,0]
                    #f_eG[:,:,ip, ic, ilr] = e_g[:,:,-1]
        
                    #sameG[0:save_neurs,:,ip, ic, ib] = e_g[0:save_neurs,:]
                e=False
                
            count+=1
        #eG= AA['arr_4']

        np.savez(stri+'losses_plots_bc.npz', losses, losses_unk, losses_kno, losses_G, losses_B)
#%%
losses[losses==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan



#%%
used_percs=percs[::2]
norm = False
Lo = losses_kno
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

fig = plt.figure(figsize=[3*3.5, 2.7])

for ib, bi in enumerate(betas):
    ax = fig.add_subplot(1, len(betas), ib+1)
    for ip, pi in enumerate(percs):
        if np.min(np.abs(pi-used_percs))==0:
            if norm:
                mm = np.nanmean((Lo[:,ip,:, ib]/Lo[0,ip,:, ib]),-1)#/np.nanmean(Lo[0,ip,:], -1)
                ss = np.nanstd((Lo[:,ip,:, ib]/Lo[0,ip,:, ib]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
            else:
                mm = np.nanmean((Lo[:,ip,:, ib]),-1)#/np.nanmean(Lo[0,ip,:], -1)
                ss = np.nanstd((Lo[:,ip,:, ib]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
            xx = np.arange(len(mm))*10
            plt.plot(xx, mm, c=palette[ip])
            plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
    ax.set_ylabel('error recorded act.')
    ax.set_xlabel('epochs')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yscale('log')
    ax.set_ylim([0.0011, 99])
    ax.set_yticks([0.01, 1,  100])
    ax.text(1500, 100, r'$\beta=$'+str(bi))
plt.savefig('Figs/F6_beta_'+str(beta)+'_singleneuron_recloss.pdf', transparent=True)

#%%
norm = False
Lo = losses_kno
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ib, bi in enumerate(betas):
    for ip, pi in enumerate(percs):

        dat = Lo[-1,ip,:, ib]
        plt.scatter(pi*np.ones(len(dat)), dat, color=palette2[ib], alpha=0.5, rasterized=True)
        plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette2[ib], zorder=4, rasterized=True)
        if ip<len(percs)-1:
            plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:,ib])], c=palette2[ib], lw=2)

ax.set_ylabel('error recorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
# if beta==2:
#     ax.set_yticks([0.001, 0.1,   10,  1000])
#     ax.set_ylim([0.001, 700])
# else:
#     ax.set_yticks([0.1,  1., 10, 100, 1000])
#     ax.set_ylim([0.005, 700])
ax.set_yscale('log')
ax.set_ylim([0.0011, 999])
ax.set_yticks([0.1, 10,  1000])
ax.set_xscale('log')
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])


plt.savefig('Figs/F6_betaall_singleneuron_rec_final.pdf', transparent=True, dpi = 300)
#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5*3, 2.7])
for ib, bi in enumerate(betas):
    ax = fig.add_subplot(1, len(betas), ib+1)

    for ip, pi in enumerate(percs):
        if np.min(np.abs(pi-used_percs))==0:
            if norm:
                mm = np.nanmean((Lo[:,ip,:,ib]/Lo[0,ip,:,ib]),-1)#/np.nanmean(Lo[0,ip,:], -1)
                ss = np.nanstd((Lo[:,ip,:,ib]/Lo[0,ip,:,ib]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
            else:
                mm = np.nanmean((Lo[:,ip,:,ib]),-1)#/np.nanmean(Lo[0,ip,:], -1)
                ss = np.nanstd((Lo[:,ip,:,ib]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
            xx = np.arange(len(mm))*10
            plt.plot(xx, mm, c=palette[ip])
            plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
    ax.set_ylabel('error unrecorded act.')
    ax.set_xlabel('epochs')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yscale('log')
    ax.set_ylim([0.0011, 999])
    ax.set_yticks([0.1, 10,  1000])
    ax.text(1500, 100, r'$\beta=$'+str(bi))
plt.savefig('Figs/F6_beta_'+str(beta)+'_singleneuron_unrecloss.pdf', transparent=True)




#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ib, bi in enumerate(betas):
    for ip, pi in enumerate(percs):
    
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
        dat = Lo[-1,ip,:,ib]
        plt.scatter(pi*np.ones(len(dat)), dat, color=palette2[ib], alpha=0.5, rasterized=True)
        plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette2[ib], zorder=4, rasterized=True)
        if ip<len(percs)-1:
            plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:,ib])], c=palette2[ib], lw=2)
        
        if ip==0:
            dat = losses_kno[0,ip,:,ib]
            pi =0.5
            plt.scatter(pi*np.ones(len(dat)), dat, color=palette2[ib], alpha=0.5, rasterized=True)
            plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette2[ib], zorder=4, rasterized=True)

    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_xscale('log')
xl = ax.get_xlim()
ax.plot(xl, [15, 15], lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')


ax.set_yscale('log')
ax.set_ylim([0.0011, 10000])
ax.set_yticks([0.1, 10,  1000])
ax.set_xscale('log')
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])


plt.savefig('Figs/F6_betaall_singleneuron_unrec_final.pdf', transparent=True)


#%%
norm = False
Lo = losses_G
palette = plt.cm.plasma(np.linspace(0, 0.8, len(used_percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(211)
for ip, pi in enumerate(used_percs):
    if norm:
        mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    else:
        mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    xx = np.arange(len(mm))*10
    plt.plot(xx, mm, c=palette[ip])
    #plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
ax.set_ylabel('error gain')
#ax.set_xlabel('epochs')
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylim([0, 1.2])
ax.set_yticks([0, 1.0])
ax.set_yticklabels(['0.0', '1.0'])
plt.savefig('Figs/F6_beta_'+str(beta)+'_singleneuron_gains.pdf', transparent=True)

norm = False
Lo = losses_B
ax = fig.add_subplot(212)
for ip, pi in enumerate(used_percs):
    if norm:
        mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    else:
        mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    xx = np.arange(len(mm))*10
    plt.plot(xx, mm, c=palette[ip])
    #plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
ax.set_ylabel('error bias')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_ylim([0, 7])
ax.set_ylim([0, 0.7])
#ax.set_yscale('log')
plt.savefig('Figs/F6_beta_'+str(beta)+'_singleneuron_biases.pdf', transparent=True)
#%%
dtype = torch.FloatTensor 
input_train, output_train, cond_train = fl.create_input(taus, trials)
input_train = torch.from_numpy(input_train).type(dtype)
output_train = torch.from_numpy(output_train).type(dtype)
# #%%

# for beta in [0.5, 2]:
# # Plot error
#     ic  = 5 #6 is good
#     IPS = np.arange(len(used_percs))#[3, 5, 6, 7, 8, 12, 14]
#     for IP in [3,7]:
    
#         tri = 2
#         print(str(used_percs[IP]) + '  '+str(beta)+'  '+str(ic))
#         AA = np.load(stri+'task'+label+'_lossesBeta'+str(beta)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
        
#         e_g = AA['arr_5']
#         e_b = AA['arr_6']
        
#         #f_net = np.load('example_net_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
#         J = f_net['arr_0']
#         gt = f_net['arr_1']
#         bt = f_net['arr_2']
#         wI = f_net['arr_3']
#         wout = f_net['arr_4']
#         x0 = f_net['arr_5']
#         mwrec_ = f_net['arr_6']
#         refEI_ = f_net['arr_7']
#         refVec = refEI[0,:]
        
#         input_size = np.shape(wI)[0]
#         N = np.shape(wI)[1]
#         hidden_size = N
        
#         gtr = e_g[:,-1]+gt
#         btr = e_b[:,-1]+bt
        
#         g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
#         g_Tra  = torch.from_numpy(gtr[:,np.newaxis]).type(dtype)
#         b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
#         b_Tra  = torch.from_numpy(btr[:,np.newaxis]).type(dtype)
        
#         wi_init = torch.from_numpy(wI).type(dtype)
#         wrec= torch.from_numpy(J).type(dtype)
#         H0 = torch.from_numpy(x0).type(dtype)
#         mwrec = torch.from_numpy(mwrec_).type(dtype)
#         refEI = torch.from_numpy(refEI_).type(dtype)
#         wo_init2 = np.eye(N)
#         wo_init2 = torch.from_numpy(wo_init2).type(dtype)
        
#         Net_True = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
#                                   train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
#         NetJ2    = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_Tra, g_Tra, h0_init=H0,
#                                   train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False, beta = beta)
#         print(g_Tra[0])
#         Out_True = Net_True(input_train)
#         out_true = Out_True.detach().numpy()
        
#         Out_Tra = NetJ2(input_train)
#         out_tra = Out_Tra.detach().numpy()
        
#         selecN = np.arange(300)
#         selecT = 3
#         vMa = 10.
#         fig = plt.figure(figsize=[1.5*2.2, 1.5*2*0.7])#[3.5, 3.2])
#         ax = fig.add_subplot(111)
#         plt.imshow(np.abs(out_tra[tri,::selecT,selecN]-out_true[tri,::selecT,selecN]), cmap='Greys', vmin=0, vmax=vMa, origin='lower', rasterized=True, aspect='auto')
#         rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor='k', alpha=0.5, facecolor='none')
#         ax.add_patch(rect)
#         rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor=palette[IP], alpha=0.9, facecolor='none')
#         ax.add_patch(rect)
#         ax.set_ylabel('neuron id')
#         ax.set_xlabel('time')
#         ax.set_xticks([0, 100])#labels('')
#         ax.set_xticklabels(['0', '15'])
#         plt.savefig('Figs/F6_beta_'+str(beta)+'_example_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
#         plt.show()
        
#         #%
#         if IP==IPS[-1]:
#             fig = plt.figure(figsize=[3.5, 3.2])
#             ax = fig.add_subplot(111)
#             plt.imshow(0*np.abs(out_tra[tri,::selecT,selecN]-out_true[tri,::selecT,selecN]), cmap='Greys', vmin=0, vmax=vMa, rasterized=True, origin='lower',)
#             cb = plt.colorbar()
#             ax.set_ylabel('neuron id', color='w')
#             ax.set_xlabel('time', color='w')
#             ax.set_xticks([0, 100])#labels('')
#             ax.set_xticklabels(['0', '15'], color='w')
#             ax.set_yticklabels([''], color='w')
            
            
#             plt.savefig('Figs/F6_beta_'+str(beta)+'_example_legend.pdf', dpi=300, transparent=True)
#             plt.show()
#         print((np.mean((out_tra[tri,::selecT,0:used_percs[IP]]-out_true[tri,::selecT,0:used_percs[IP]])**2)))
#         print((np.mean((out_tra[tri,::selecT,used_percs[IP]:]-out_true[tri,::selecT,used_percs[IP]:])**2)))
#         print('')
#         #%
#         # Find trials
#         TRs = []
#         inpTR = []
#         for itrr in range(trials):
#             blpr = input_train[itrr,:,:].sum(axis=0).detach().numpy()
    
#             if blpr[0] == 0. and blpr[1] == 1. and len(TRs)==0:
#                 TRs.append([itrr])
#                 inpTR.append([np.argmax(input_train[itrr,:,1].detach().numpy())])
#             elif len(TRs)==1 and blpr[0] == 1.:
#                 TRs.append([itrr])
#                 inpTR.append([np.argmax(input_train[itrr,:,0].detach().numpy())])
#             elif len(TRs)==2 and np.sum(blpr) == 0.:
#                 TRs.append([itrr])
#             else:
#                 continue
#         reads_true = np.zeros((len(TRs), np.shape(out_true)[1]))
#         reads_tra = np.zeros((len(TRs), np.shape(out_true)[1]))
        
#         for itR, tR in enumerate(TRs):
#             reads_true[itR,:] = np.dot(wout, out_true[tR[0],:,:].T)
#             reads_tra[itR,:] = np.dot(wout, out_tra[tR[0],:,:].T)

#         #%%
#         fig = plt.figure(figsize=[1.5*2.2, 1.5*2*0.7])
#         ax = fig.add_subplot(111)
#         plt.plot(taus, reads_true[0,:],  '--', c='k',lw=0.8)
#         plt.plot(taus, reads_tra[0,:], lw=1.5, color=palette[IP])
#         plt.xlabel('time')
#         plt.ylabel('readout')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         ax.set_yticks([-1, 0, 1])
#         plt.savefig('Figs/F6_beta_'+str(beta)+'_readout_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
#         plt.show()


#%%
# bbetas = [0.25, 0.5, 1., 2., 4.]
# for ic in [2,3,4,5,6,7]:
#      #5 and 6 is good
#     IPS = np.arange(len(used_percs))#[3, 5, 6, 7, 8, 12, 14]
#     for IP in [3, 7]:#IPS:
#         fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
#         ax = fig.add_subplot(111)
#         for ibeta, beta in enumerate(bbetas):#betas):
#             # Plot error
#             # if beta>1:
#             #     beta=int(beta)
        
#             tri = 2
#             print(str(used_percs[IP]) + '  '+str(beta)+'  '+str(ic))
#             AA = np.load(stri+'task'+label+'_lossesBeta'+str(beta)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
            
#             e_g = AA['arr_5']
#             e_b = AA['arr_6']
            
#             #f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
#             J = f_net['arr_0']
#             gt = f_net['arr_1']
#             bt = f_net['arr_2']
#             wI = f_net['arr_3']
#             wout = f_net['arr_4']
#             x0 = f_net['arr_5']
#             mwrec_ = f_net['arr_6']
#             refEI_ = f_net['arr_7']
#             refVec = refEI[0,:]
            
#             input_size = np.shape(wI)[0]
#             N = np.shape(wI)[1]
#             hidden_size = N
            
#             gtr = e_g[:,-1]+gt
#             btr = e_b[:,-1]+bt
            
#             g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
#             g_Tra  = torch.from_numpy(gtr[:,np.newaxis]).type(dtype)
#             b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
#             b_Tra  = torch.from_numpy(btr[:,np.newaxis]).type(dtype)
            
#             wi_init = torch.from_numpy(wI).type(dtype)
#             wrec= torch.from_numpy(J).type(dtype)
#             H0 = torch.from_numpy(x0).type(dtype)
#             mwrec = torch.from_numpy(mwrec_).type(dtype)
#             refEI = torch.from_numpy(refEI_).type(dtype)
#             wo_init2 = np.eye(N)
#             wo_init2 = torch.from_numpy(wo_init2).type(dtype)
            
#             Net_True = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
#                                       train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
            
#             NetJ2    = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_Tra, g_Tra, h0_init=H0,
#                                       train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False, beta = beta)
#             Out_True = Net_True(input_train)
#             out_true = Out_True.detach().numpy()
            
#             Out_Tra = NetJ2(input_train)
#             out_tra = Out_Tra.detach().numpy()
            
#             selecN = np.arange(300)
#             selecT = 3
#             vMa = 10.

#             print((np.mean((out_tra[tri,::selecT,0:used_percs[IP]]-out_true[tri,::selecT,0:used_percs[IP]])**2)))
#             print((np.mean((out_tra[tri,::selecT,used_percs[IP]:]-out_true[tri,::selecT,used_percs[IP]:])**2)))
#             print('')
#             III  = 0#used_percs[IP]-1
#             mse = np.mean((out_tra[tri,::selecT,III]-out_true[tri,::selecT,III])**2)

#             if ibeta==0:
#                 plt.plot(taus[::selecT], out_true[tri,::selecT, III], c='k', lw=3, alpha=0.5, label='teacher')
#                 plt.plot(taus[::selecT], out_tra[tri,::selecT, III], color=palette2[ibeta], lw=1.5, label=r'$\beta=$'+str(beta) + ' MSE ='+"{:.2e}".format(mse))
#             else:
#                 plt.plot(taus[::selecT], out_tra[tri,::selecT, III], color=palette2[ibeta], lw=1.5, label=r'$\beta=$'+str(beta) + '   MSE ='+"{:.2e}".format(mse))
    
#         plt.xlabel('time')
#         plt.ylabel('rec. neuron activity')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         plt.savefig('Figs/recfigSupp_nrec_'+str(used_percs[IP])+'_ic_'+str(ic)+'.pdf')

# #%%

# for ic in [2,3,4,5,6,7]:
#      #5 and 6 is good
#     IPS = np.arange(len(used_percs))#[3, 5, 6, 7, 8, 12, 14]
#     for IP in [3, 7]:#IPS:
#         fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
#         ax = fig.add_subplot(111)
#         for ibeta, beta in enumerate(betas):
#             # Plot error
        
        
#             tri = 2
#             print(used_percs[IP])
#             AA = np.load(stri+'task'+label+'_lossesBeta'+str(beta)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
            
#             e_g = AA['arr_5']
#             e_b = AA['arr_6']
            
#             #f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
#             J = f_net['arr_0']
#             gt = f_net['arr_1']
#             bt = f_net['arr_2']
#             wI = f_net['arr_3']
#             wout = f_net['arr_4']
#             x0 = f_net['arr_5']
#             mwrec_ = f_net['arr_6']
#             refEI_ = f_net['arr_7']
#             refVec = refEI[0,:]
            
#             input_size = np.shape(wI)[0]
#             N = np.shape(wI)[1]
#             hidden_size = N
            
#             gtr = e_g[:,-1]+gt
#             btr = e_b[:,-1]+bt
            
#             g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
#             g_Tra  = torch.from_numpy(gtr[:,np.newaxis]).type(dtype)
#             b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
#             b_Tra  = torch.from_numpy(btr[:,np.newaxis]).type(dtype)
            
#             wi_init = torch.from_numpy(wI).type(dtype)
#             wrec= torch.from_numpy(J).type(dtype)
#             H0 = torch.from_numpy(x0).type(dtype)
#             mwrec = torch.from_numpy(mwrec_).type(dtype)
#             refEI = torch.from_numpy(refEI_).type(dtype)
#             wo_init2 = np.eye(N)
#             wo_init2 = torch.from_numpy(wo_init2).type(dtype)
            
#             Net_True = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
#                                       train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
            
#             NetJ2    = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_Tra, g_Tra, h0_init=H0,
#                                       train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False, beta = beta)
            
#             Out_True = Net_True(input_train)
#             out_true = Out_True.detach().numpy()
            
#             Out_Tra = NetJ2(input_train)
#             out_tra = Out_Tra.detach().numpy()
            
#             selecN = np.arange(300)
#             selecT = 3
#             vMa = 10.
            
#             print(np.mean((out_tra[tri,::selecT,0:used_percs[IP]]-out_true[tri,::selecT,0:used_percs[IP]])**2))
#             print(np.mean((out_tra[tri,::selecT,used_percs[IP]:]-out_true[tri,::selecT,used_percs[IP]:])**2))
#             print('')
#             III = 150#used_percs[IP]
#             mse = np.mean((out_tra[tri,::selecT,III]-out_true[tri,::selecT,III])**2)
#             if ibeta==0:
#                 plt.plot(taus[::selecT], out_true[tri,::selecT,III], c='k', lw=3, alpha=0.5, label='teacher')
#                 plt.plot(taus[::selecT], out_tra[tri,::selecT,III], color=palette2[ibeta], lw=1.5, label=r'$\beta=$'+str(beta) + ' MSE ='+"{:.2e}".format(mse))
#             else:
#                 plt.plot(taus[::selecT], out_tra[tri,::selecT,III], color=palette2[ibeta], lw=1.5, label=r'$\beta=$'+str(beta) + '   MSE ='+"{:.2e}".format(mse))
    
#         plt.xlabel('time')
#         plt.ylabel('unrec. neuron activity')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         plt.savefig('Figs/figSupp_nrec_'+str(used_percs[IP])+'_ic_'+str(ic)+'.pdf')
#         plt.show()
    


