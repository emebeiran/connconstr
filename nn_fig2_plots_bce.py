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


#%%
N = 300
dta = 0.05
taus = np.arange(0, 20, dta)
Nt = len(taus)
Tau = 0.2


trials = 60
stri = 'Data/Figure2/' 

seed =21
np.random.seed(seed)

#generate trials
input_train, output_train, cond_train = fl.create_input(taus, trials)



label = '_random'#''

#%%
f_net = np.load(stri+'teacherRNN_cycling.npz')
label=''
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

allseeds = 10#int(sys.argv[2])
percs = np.array((1, 2, 4, 8,10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 150, 200, 250,)) #300))


#%%
ics = 10

losses_ = []
eG_ = []

losses = []
eG = []


save_neurs = 100

count = 0
used_percs = percs#[0:16]
e=False
dorandom =True

lr=0.001#lr = 0.005
dogain=True
try:
    #np.savez(stri+'losses_fig2_bce.npz', losses, losses_unk, losses_kno, losses_unk_R, losses_kno_R, 
    #         losses_G, losses_B, losses_Grec, losses_Brec, losses_Gunr, losses_Bunr, meG, sameG,)
    bb = np.load(stri+'losses_fig2_bce.npz')
    losses = bb['arr_0']
    losses_unk = bb['arr_1']
    losses_kno = bb['arr_2']
    losses_unk_R = bb['arr_3']
    losses_kno_R = bb['arr_4']
    losses_G = bb['arr_5']
    losses_B = bb['arr_6']
    losses_Grec = bb['arr_7']
    losses_Brec = bb['arr_8']
    losses_Gunr = bb['arr_9']
    losses_Bunr = bb['arr_10']
    meG = bb['arr_11']
    sameG = bb['arr_12']
except:
    print('loading datafiles for losses')
    for ic in range(ics):
        for ip in range(len(percs)):
            try:
                if dogain:#gainRlosses is for random selection, gainlosses is always the same
                    if dorandom:
                        #AA = np.load(stri+'task'+label+'_gainRsllosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                        AA = np.load(stri+'task'+label+'_gainRslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                    else:
                        AA = np.load(stri+'task'+label+'_gainlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                    
                else:
                    AA = np.load(stri+'task'+label+'_losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                if dorandom:
                    seq = AA['arr_9']
                else:
                    seq = np.arange(N)
                losses_ = AA['arr_0']
                #losses_read_ = AA['arr_1']
                losses_unk_ = AA['arr_1']
                losses_kno_ = AA['arr_2']
                losses_G_ = AA['arr_3']
                losses_B_ = AA['arr_4']
                e_g = AA['arr_5']
                e_b = AA['arr_6']
                losses_unk_R_ = AA['arr_10']#This are CV
                losses_kno_R_ = AA['arr_11']     #This are CV       
                if len(losses_)!=700:
                    a = RaiseError[0]
            except:
                print('ic '+str(ic))
                print('nRec '+str(used_percs[ip]))
                print('error --')
                e=True

            
            if count==0:
                losses = np.zeros((len(losses_), len(used_percs), ics ))
                losses_unk = np.zeros((len(losses_unk_), len(used_percs), (ics) ))
                losses_kno = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                losses_unk_R = np.zeros((len(losses_unk_), len(used_percs), (ics) ))
                losses_kno_R = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                losses_G = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                losses_B = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                losses_Grec = np.zeros((len(losses_kno_)//10, len(used_percs), (ics) ))
                losses_Brec = np.zeros((len(losses_kno_)//10, len(used_percs), (ics) ))
                losses_Gunr = np.zeros((len(losses_kno_)//10, len(used_percs), (ics) ))
                losses_Bunr = np.zeros((len(losses_kno_)//10, len(used_percs), (ics) ))  
                meG = np.zeros((np.shape(e_g)[1], len(used_percs), (ics) ))
                sameG = np.zeros((save_neurs, np.shape(e_g)[1], len(used_percs), (ics) ))
                
                #i_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                #f_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
            if e:
                losses[:,ip, ic] = np.nan
                losses_unk[:,ip, ic] = np.nan
                losses_kno[:,ip, ic] = np.nan
                losses_unk_R[:,ip, ic] = np.nan
                losses_kno_R[:,ip, ic] = np.nan
                losses_G[:,ip, ic] = np.nan
                losses_B[:,ip, ic] = np.nan
                losses_Grec[:,ip, ic] = np.nan
                losses_Brec[:,ip, ic] = np.nan
                losses_Gunr[:,ip, ic] = np.nan
                losses_Bunr[:,ip, ic] = np.nan
                meG[:,ip, ic] = np.nan
                #i_eG[:,:,ip, ic, ilr] = np.nan
                #f_eG[:,:,ip, ic, ilr] = np.nan
                #sameG[:,ip, ic, ilr] = np.nan
                
            else:
                losses[:,ip, ic] = losses_
                losses_unk[:,ip, ic] = losses_unk_
                losses_kno[:,ip, ic] = losses_kno_
                losses_unk_R[:,ip, ic] = losses_unk_R_
                losses_kno_R[:,ip, ic] = losses_kno_R_
                losses_G[:,ip, ic] = losses_G_
                losses_B[:,ip, ic] = losses_B_
                
                losses_Grec[:,ip, ic] = np.sqrt(np.mean(e_g[seq[0:used_percs[ip]],:]**2, 0))
                losses_Brec[:,ip, ic] = np.sqrt(np.mean(e_b[seq[0:used_percs[ip]],:]**2, 0))
                
                losses_Gunr[:,ip, ic] = np.sqrt(np.mean(e_g[seq[used_percs[ip]:],:]**2, 0))
                losses_Bunr[:,ip, ic] = np.sqrt(np.mean(e_b[seq[used_percs[ip]:],:]**2, 0))
                
                meG[:,ip, ic] = np.sqrt(np.mean(e_g**2,axis=(0)))
                #i_eG[:,:,ip, ic, ilr] = e_g[:,:,0]
                #f_eG[:,:,ip, ic, ilr] = e_g[:,:,-1]

                sameG[0:save_neurs,:,ip, ic] = e_g[0:save_neurs,:]
            e=False
            
        count+=1
        #eG= AA['arr_4']
    np.savez(stri+'losses_fig2_bce.npz', losses, losses_unk, losses_kno, losses_unk_R, losses_kno_R, 
            losses_G, losses_B, losses_Grec, losses_Brec, losses_Gunr, losses_Bunr, meG, sameG,)
#%%
losses[losses==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan
losses_unk_R[losses_unk_R==0] = np.nan
losses_kno_R[losses_kno==0] = np.nan

#%%
try:
    AA = np.load(stri+'statistic_baseline.npz')
    sols = AA['arr_0']
    sols2 = AA['arr_1']
    sols_N = AA['arr_2']
    sols2_N = AA['arr_3']

except:
    print('calculating baselines')
    
    used_percs = percs#[::2]#[1::2]
    # Plot error
    ic  = 5 #6 is good
    IPS = np.arange(len(used_percs))#[3, 5, 6, 7, 8, 12, 14]


    palette = plt.cm.plasma(np.linspace(0, 0.8, len(used_percs)))
    IP =0

    tri = 2
    print(used_percs[IP])
    if dogain:
        if dorandom:
            AA = np.load(stri+'task'+label+'_gainRsllosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
        else:
            AA = np.load(stri+'task'+label+'_gainlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
        
    else:
        AA = np.load(stri+'task'+label+'_losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')

    e_g = AA['arr_5']
    e_b = AA['arr_6']

    #f_net = np.load('example_net_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0

    if dogain:
        f_net = np.load(stri+'teacherRNN_cycling.npz')#np.load('example_netGain_softplus_2.npz')
        
    else:
        f_net = np.load('example_net_softplus_2.npz')
        

    dtype = torch.FloatTensor 
    input_train, output_train, cond_train = fl.create_input(taus, trials)
    input_train = torch.from_numpy(input_train).type(dtype)
    output_train = torch.from_numpy(output_train).type(dtype)

    J = f_net['arr_0']
    gt = f_net['arr_1']
    bt = f_net['arr_2']
    wI = f_net['arr_3']
    wout = f_net['arr_4']
    x0 = f_net['arr_5']
    mwrec_ = f_net['arr_6']
    refEI_ = f_net['arr_7']
    refVec = refEI[0,:]

    if dorandom:
        seq = AA['arr_9']
        J = J[seq,:]
        J = J[:,seq]

        gt = gt[seq]
        bt = bt[seq]

        wI = wI[:,seq]
        wout = wout[seq]
        x0 = x0[seq]
        mwrec_ = mwrec_[seq,:]
        mwrec_ = mwrec_[:,seq]
        refEI_ = refEI_[seq,:]
        refEI_ = refEI_[:,seq]
        refVec = refVec[seq]


    input_size = np.shape(wI)[0]
    N = np.shape(wI)[1]
    hidden_size = N

    g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
    b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)

    wi_init = torch.from_numpy(wI).type(dtype)
    wrec= torch.from_numpy(J).type(dtype)
    H0 = torch.from_numpy(x0).type(dtype)
    mwrec = torch.from_numpy(mwrec_).type(dtype)
    refEI = torch.from_numpy(refEI_).type(dtype)
    wo_init2 = np.eye(N)
    wo_init2 = torch.from_numpy(wo_init2).type(dtype)
    if dogain:
        Net_True = fl.RNNgain(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                                  train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)

    else:
        Net_True = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                                  train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        

    Out_True = Net_True(input_train)
    out_true = Out_True.detach().numpy()

    pers = 50
    
    sols = np.zeros(pers)
    sols2 = np.zeros(pers)
    
    sols_N = np.zeros((pers,N-1))
    sols2_N = np.zeros((pers,N-1))
    
    outtN= out_true.reshape(-1, out_true.shape[-1])
        
    for ipe in range(pers): 
        print('pert: '+str(ipe))
        out_perm = out_true[:,:,np.random.permutation(N)]
        sol = np.mean((out_true-out_perm)**2)
    
        sols[ipe] = sol
        outsN= out_perm.reshape(-1, out_perm.shape[-1])
    
        CC = np.corrcoef(outsN.T, outtN.T)
                # CVs = np.zeros(N)
        CVs = np.diag(CC[0:N,N:])
        sols2[ipe]=1-np.mean(np.abs(CVs))
        for i in range(N-1):
            print(i)
            sols_N[ipe, i]=sol
            sols2_N[ipe, i]=1-np.mean(np.abs(CVs))
            
            out_perm_ = out_perm[:,:,i:]
            out_true_ = out_true[:,:,i:]
            sol = np.mean((out_true_-out_perm_)**2)
            
            
            outsN= out_perm_.reshape(-1, out_perm_.shape[-1])
            outtN_= out_true_.reshape(-1, out_true_.shape[-1])
            CC = np.corrcoef(outsN.T, outtN_.T)
                    # CVs = np.zeros(N)
            CVs = np.diag(CC[0:N,N:])
            

    np.savez(stri+'statistic_baseline.npz', sols, sols2, sols_N, sols2_N)
#%%
fl.set_plot()

#%%




used_percs=percs[::2]
norm = False
Lo = losses_kno
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    if np.min(np.abs(pi-used_percs))==0:
        if norm:
            mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
        else:
            mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
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
ax.set_yticks([0.001, 0.1, 10])
#ax.set_yticklabels([1e-3, '0.1', '10'])
ax.set_ylim([0.001, 50])
plt.savefig('Figs/F1_singleneuron_recloss.pdf', transparent=True, dpi=250)

#%%
norm = False
Lo = losses_kno
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
    dat = Lo[-1,ip,:]
    plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.8, rasterized=True)
    plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette[ip], zorder=4, rasterized=True)
    if ip<len(percs)-1:
        plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
#ax.set_ylabel('error recorded act.')
ax.set_xlabel('rec. units M')
ax.set_ylabel('error recorded act.')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
ax.set_yticks([0.001, 0.1, 10])
#ax.set_yticklabels([1e-3, '0.1', '10'])
ax.set_ylim([0.001, 50])
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])
ax.set_xscale('log')
#plt.savefig('Figs/F1_singleneuron_rec_final.pdf')#, transparent=True
#%%

E_all = np.mean(sols)
E_CV_all  = np.mean(sols2)

norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    if np.min(np.abs(pi-used_percs))==0:
        if norm:
            mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
        else:
            mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
        xx = np.arange(len(mm))*10
        plt.plot(xx, mm, c=palette[ip])
        plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*E_all, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')

ax.set_yticks([0.1, 1, 10])
#ax.set_yticklabels([0.1, 1, 10])
ax.set_ylim([0.01, 50])

#plt.savefig('Figs/F1_singleneuron_unrecloss.pdf', transparent=True)

#%%

norm = False
Lo = losses_unk_R
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    if np.min(np.abs(pi-used_percs))==0:
        if norm:
            mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
        else:
            mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
        xx = np.arange(len(mm))*10
        plt.plot(xx, mm, c=palette[ip])
        plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
        
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*E_CV_all, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
ax.set_yticks([0.001, 0.1, 10])
#ax.set_yticklabels([1e-3, '0.1', '10'])
ax.set_ylim([0.001, 50])
# ax.set_xticks([1, 10, 100])
# ax.set_xticklabels(['1', '10', '100'])
#plt.savefig('Figs/F1_singleneuron_unrecloss_CV.pdf', transparent=True)
#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
    dat = Lo[-1,ip,:]
    plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5, rasterized=True)
    plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette[ip], zorder=4, rasterized=True)
    if ip<len(percs)-1:
        plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
#ax.set_ylabel('error unrecorded act.')
ax.set_xscale('log')
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*E_all, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel('error unrecorded act.')
ax.set_yscale('log')

ax.set_yticks([0.1,  1, 10])
#ax.set_yticklabels([0.1, 0.2, 1, 10])
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])

ax.set_ylim([0.01, 50])
# ax.set_xticks([0, 50, 100, 150, 200])
#plt.savefig('Figs/F1_singleneuron_unrec_final.pdf', transparent=True)

#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
    dat = Lo[-1,ip,:]
    plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5, rasterized=True)
    plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette[ip], zorder=4, rasterized=True)
    if ip<len(percs)-1:
        plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
#ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel('error unrecorded act.')
ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_yticks([0.1,  1, 10])
#ax.set_yticklabels([0.1, 0.2, 1, 10])
# ax.set_xticks([1, 10, 100])
# ax.set_xticklabels(['1', '10', '100'])

ax.set_ylim([0.01, 50])
# ax.set_xticks([0, 50, 100, 150, 200])
#plt.savefig('Figs/F1_singleneuron_unrec_final_nolog.pdf', transparent=True)

#%%
norm = False
Lo = losses_unk_R
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[2.0, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
    dat = Lo[-1,ip,:]
    plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5)
    plt.scatter(pi, np.nanmean(dat), s=30, edgecolor='k', color=palette[ip], zorder=4)
    if ip<len(percs)-1:
        plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
#ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_yticks([0.001, 0.01, 0.1, 1])
# ax.set_xticks([0, 50, 100, 150, 200])
#plt.savefig('Figs/F1_singleneuron_unrecCV_final.pdf', transparent=True)



#%%

pers = 50
e_g_=np.zeros(pers)
e_b_=np.zeros(pers)

for ipe in range(pers):
    e_g_[ipe] = np.sqrt(np.mean((gt-np.random.permutation(gt))**2))
    e_b_[ipe] = np.sqrt(np.mean((bt-np.random.permutation(bt))**2))
    
Eg = np.mean(e_g_)
Eb = np.mean(e_b_)

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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error gain')
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*Eg, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
#ax.set_xlabel('epochs')
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylim([0, 1.2])
ax.set_yticks([0, 1.0])
ax.set_yticklabels(['0.0', '1.0'])
plt.savefig('Figs/F1_singleneuron_gains.pdf', transparent=True)

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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error bias')
ax.set_xlabel('epochs')
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*Eb, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_ylim([0, 7])
ax.set_ylim([0, 0.7])
#ax.set_yscale('log')
#plt.savefig('Figs/F1_singleneuron_biases.pdf', transparent=True)

#%%
norm = False
Lo = losses_Grec
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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error gain')
#ax.set_xlabel('epochs')
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylim([0, 0.8])
ax.set_yticks([0, .8])
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*Eg, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
#ax.set_yticklabels(['0.0', '1.0'])
#plt.savefig('Figs/F1_singleneuron_gains_rec.pdf', transparent=True)

norm = False
Lo = losses_Brec
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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error bias')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*Eb, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
#ax.set_ylim([0, 7])
ax.set_ylim([0, 0.5])
ax.set_yticks([0, .5])
#ax.set_yscale('log')
#plt.savefig('Figs/F1_singleneuron_biases_rec.pdf', transparent=True)

#%%
norm = False
Lo = losses_Gunr
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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error gain')
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*Eg, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
#ax.set_xlabel('epochs')
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylim([0, 0.8])
ax.set_yticks([0, .8])
#ax.set_yticklabels(['0.0', '1.0'])
#plt.savefig('Figs/F1_singleneuron_gains_unrec.pdf', transparent=True)

norm = False
Lo = losses_Bunr
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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
xl = ax.get_xlim()        
ax.plot(xl, np.ones_like(xl)*Eb, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.set_ylabel('error bias')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_ylim([0, 7])
ax.set_ylim([0, 0.5])
ax.set_yticks([0, .5])
#ax.set_yscale('log')
#plt.savefig('Figs/F1_singleneuron_biases_unrec.pdf', transparent=True)

#%%
norm = False
Lo = losses_Gunr
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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
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
plt.savefig('Figs/F1_singleneuron_gains_unrec.pdf', transparent=True)

norm = False
Lo = losses_Bunr
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
    plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error bias')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_ylim([0, 7])
ax.set_ylim([0, 0.7])
#ax.set_yscale('log')
#plt.savefig('Figs/F1_singleneuron_biases_unrec.pdf', transparent=True)