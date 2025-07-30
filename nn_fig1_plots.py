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
stri = 'Data/Figure1/' 

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
refEI_ = f_net['arr_7']
refVec = refEI_[0,:]

alpha = dta


#%%
# Get baselines
print('getting baselines')
pert =100
err_gt = np.zeros(pert)
err_bt = np.zeros(pert)
err_act =  np.zeros(pert)
err_read = np.zeros(pert)


dtype = torch.FloatTensor 
input_train, output_train, cond_train = fl.create_input(taus, trials)
input_train = torch.from_numpy(input_train).type(dtype)
output_train = torch.from_numpy(output_train).type(dtype)

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

Net_True = fl.RNNgain(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                              train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)

Out_True = Net_True(input_train)
out_true = Out_True.detach().numpy()
read_true = out_true.dot(wout[:,np.newaxis])
for ip in range(pert):
    gp = np.random.permutation(gt)
    err_gt[ip] = np.sqrt(np.mean((gp-gt)**2))
    bp = np.random.permutation(bt)
    err_bt[ip] = np.sqrt(np.mean((bt-bp)**2))
    out_pert = out_true[:,:,np.random.permutation(N)]
    #read_pert = out_pert.dot(wout[:,np.newaxis])
    err_act[ip] = np.mean((out_pert-out_true)**2)
    #err_read[ip] = np.mean((read_pert-read_true)**2)
print('end baselines')
#%%

percs = np.array((1,))#, 2, 4, 8,10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 150, 200, 250,)) #300))

ics = 30

losses_ = []
eG_ = []

losses = []
eG = []


save_neurs = 100

count = 0
used_percs = percs#[0:16]
e=False
dorandom =True

if label=='':
    lr=0.001#lr = 0.005

else:
    lr = 0.005

try:
    bbb = np.load(stri+'losses_task_Readgain3RslCV.npz')
    losses = bbb['arr_0']
    losses_unk = bbb['arr_1']
    losses_kno = bbb['arr_2']
    losses_unk_R = bbb['arr_3']
    losses_kno_R = bbb['arr_4']
    losses_G = bbb['arr_5']
    losses_B = bbb['arr_6']
    losses_Grec = bbb['arr_7']
    losses_Brec = bbb['arr_8']
    losses_Gunr = bbb['arr_9']
    losses_Bunr = bbb['arr_10']
    meG = bbb['arr_11']
    sameG = bbb['arr_12']

except:
    print('data not loaded')
    for ic in range(ics):
        for ip in range(len(percs)):
            try:

                Nal = 1
                AA = np.load(stri+'task_Readgain3RslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz')
                if dorandom:
                    seq = AA['arr_9']
                else:
                    seq = np.arange(N)
                losses_ = AA['arr_0']
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
                print('error')
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
                print(losses_kno_[-1])
                print(losses_[-1])
                print('--')
                
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
    np.savez(stri+'losses_task_Readgain3RslCV.npz', losses, losses_unk, losses_kno, losses_unk_R, losses_kno_R,
             losses_G, losses_B, losses_Grec, losses_Brec, losses_Gunr, losses_Bunr, meG, sameG)
#%%
losses[losses==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan
losses_unk_R[losses_unk_R==0] = np.nan
losses_kno_R[losses_kno==0] = np.nan

#%%
try:
    bbb = np.load(stri+'output_task_Readgain3.npz')
    outs = bbb['arr_0']
    reads = bbb['arr_1']
    valid = bbb['arr_2']
except:
    print('data not loaded')
    outs=[]
    reads=[]

    for ic in range(ics):
        print(ic)
        
        AA = np.load(stri+'task_Readgain3RslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz')
        if dorandom:
            seq = AA['arr_9']
        else:
            seq = np.arange(N)
            
        inv_seq = np.zeros(N, dtype=int)
        for i in range(N):
            inv_seq[i] = int(np.where(seq==i)[0][0])
        losses_ = AA['arr_0']
        #losses_read_ = AA['arr_1']
        losses_unk_ = AA['arr_1']
        losses_kno_ = AA['arr_2']
        losses_G_ = AA['arr_3']
        losses_B_ = AA['arr_4']
        e_g = AA['arr_5']
        e_b = AA['arr_6']
        e_gf = AA['arr_7']
        e_bf = AA['arr_8']
        
        losses_unk_R_ = AA['arr_10']#This are CV
        losses_kno_R_ = AA['arr_11']     #This are CV   
        gN = torch.from_numpy(fl.relu(gt[:,np.newaxis])+fl.relu(e_gf[inv_seq,np.newaxis])).type(dtype)
        bN = torch.from_numpy(bt[:,np.newaxis]+e_bf[inv_seq,np.newaxis]).type(dtype)
        
        Net_Ic = fl.RNNgain(input_size, hidden_size, 1, wi_init, wo_init2[:,[0,]], wrec, mwrec, refEI, bN, gN, h0_init=H0,
                                    train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        Net_Ic.load_state_dict(torch.load(stri+"net_fig0_N_"+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
        Net_Ic2 = fl.RNNgain(input_size, hidden_size, N, Net_Ic.wi, wo_init2, Net_Ic.wrec, Net_Ic.mwrec, Net_Ic.refEI, Net_Ic.b, Net_Ic.g, h0_init=Net_Ic.h0,
                                    train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
        Out_True = Net_Ic2(input_train)
        wout = Net_Ic.wout.detach().numpy()
        out_ic= Out_True.detach().numpy()
        read_ic = out_ic.dot(wout)
        reads.append(read_ic)
        outs.append(out_ic[:,:,inv_seq])
        #print(wout[inv_seq][0])
    
    

    valid =np.zeros(ics)
    for ic in range(ics):
        print(np.mean(losses_kno[:,0,ic]))
        print(np.max(outs[ic]))
        if np.max(outs[ic])<400 and ~np.isnan(np.mean(losses_kno[:,0,ic])):
            valid[ic]=True
    valid = valid==1
    np.savez(stri+'output_task_Readgain3.npz', outs[:,:,0:20], reads[:,:,0:20], valid)
#%%
fl.set_plot()

#%%
used_percs=percs[::2]
norm = False
Lo = losses_kno
palette = plt.cm.Oranges(np.random.permutation(np.linspace(0.3, 0.8, ics)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    if np.min(np.abs(pi-used_percs))==0:
        # if norm:
        #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
        # else:
        #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
        # xx = np.arange(len(mm))*10
        # plt.plot(xx, mm, c=palette[ip])
        # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
        xx = np.arange(len(Lo[:,ip,0]))*10
        for ic in range(ics):
            if valid[ic]:
                plt.plot(xx, Lo[:,ip,ic], color=palette[ic])
                print(Lo[-1,ip,ic])
ax.set_ylabel('error in readout (loss)')
ax.set_xlabel('training epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
plt.savefig('Figs/F0_error_read.pdf', transparent=True, dpi=250)

#%%
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
norm = False
Lo = losses_unk

Ics = 20
for ip, pi in enumerate(percs):
    if np.min(np.abs(pi-used_percs))==0:
        
        xx = np.arange(len(Lo[:,ip,0]))*10
        for ic in range(Ics):
            if valid[ic]:
                plt.plot(xx, Lo[:,ip,ic], color=palette[ic])
plt.plot(xx, np.mean(err_act)*np.ones(len(xx)), c='k', alpha=0.4, lw=3)
plt.ylim([0.01, 2000])
ax.set_ylabel('error in activity')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')

plt.savefig('Figs/F0_error_act.pdf', transparent=True)


#%%
norm = False
Lo = losses_G
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(211)
for ip, pi in enumerate(used_percs):
    # if norm:
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    xx = np.arange(len(Lo[:,ip,0]))*10
    for ic in range(ics):
        if valid[ic]:
            plt.plot(xx, Lo[:,ip,ic], color=palette[ic])
plt.plot(xx, np.median(err_gt)*np.ones(len(xx)), c='k', alpha=0.4, lw=3)

ax.set_ylabel('error gain')
#ax.set_xlabel('epochs')
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylim([0, 1.])
ax.set_yticks([0, 0.5])
norm = False
Lo = losses_B
ax = fig.add_subplot(212)
for ip, pi in enumerate(used_percs):
    xx = np.arange(len(Lo[:,ip,0]))*10
    for ic in range(ics):
        if valid[ic]:
            plt.plot(xx, Lo[:,ip,ic], color=palette[ic])
plt.plot(xx, np.median(err_bt)*np.ones(len(xx)), c='k', alpha=0.4, lw=3)

ax.set_ylabel('error bias')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_ylim([0, 7])
ax.set_ylim([0, 0.6])
#ax.set_yscale('log')
plt.savefig('Figs/F0_error_params.pdf', transparent=True)

#%%

# Find trials
TRs = []
inpTR = []
for itrr in range(trials):
    blpr = input_train[itrr,:,:].sum(axis=0).detach().numpy()

    if blpr[0] == 0. and blpr[1] == 1. and len(TRs)==0:
        TRs.append([itrr])
        inpTR.append([np.argmax(input_train[itrr,:,1].detach().numpy())])
    elif len(TRs)==1 and blpr[0] == 1.:
        TRs.append([itrr])
        inpTR.append([np.argmax(input_train[itrr,:,0].detach().numpy())])
    elif len(TRs)==2 and np.sum(blpr) == 0.:
        TRs.append([itrr])
    else:
        continue

#%%
IT =TRs[0][0]
NN = -16

fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)

for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)

ax.set_ylim([0, 11])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('neural activity')
ll = ax.get_ylim()
plt.savefig('Figs/F0_neuroninh_inp1.pdf', transparent=True)

#%%
IT =TRs[1][0]
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)

for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)
#ax.set_ylim([0, 18])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('neural activity')
ax.set_ylim(ll)
plt.savefig('Figs/F0_neuroninh_inp2.pdf', transparent=True)

#%%
IT =TRs[1][0]
NN = -16

fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(211)

for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)

ax.set_ylim([0, 11])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('aa', color='w')
ll = ax.get_ylim()
IT =TRs[0][0]
ax.set_xlabel('')
ax.set_xticklabels('')
ax = fig.add_subplot(212)
for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)
#ax.set_ylim([0, 18])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('aa', color='w')
ax.set_ylim(ll)
plt.savefig('Figs/F0_neuroninh_inpBoth.pdf', transparent=True)

#%%
IT =TRs[0][0]
NN = 25
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)

for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('neural activity')
ax.set_ylim([0, 24])
plt.savefig('Figs/F0_neuronexc_inp1.pdf', transparent=True)

#%%
IT =TRs[1][0]
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('neural activity')
ax.set_ylim([0, 24])
plt.savefig('Figs/F0_neuronexc_inp2.pdf', transparent=True)

#%%
IT =TRs[1][0]
NN = 25

fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(211)

for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)

ax.set_ylim([0, 24])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('aa', color='w')
ll = ax.get_ylim()
IT =TRs[0][0]
ax.set_xlabel('')
ax.set_xticklabels('')
ax = fig.add_subplot(212)
for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, outs[ic][IT,:,NN], color=palette[ic])
plt.plot(taus, out_true[IT,:,NN], c='k', lw=1.5, alpha=0.8)
#ax.set_ylim([0, 18])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('aa', color='w')
ax.set_ylim(ll)
plt.savefig('Figs/F0_neuronexc_inpBoth.pdf', transparent=True)


#%%
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)

for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, reads[ic][5,:], color=palette[ic])
plt.plot(taus, read_true[5,:], c='k', lw=1.5, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('readout')
plt.ylim([-1.5, 1.5])

plt.savefig('Figs/F0_readout.pdf', transparent=True)

#%%
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(211)
IT =TRs[1][0]
count = 0
plt.plot(taus, read_true[IT,:], c='k', lw=1.5, alpha=0.8, label='teacher')
for ic in range(ics):
    if valid[ic]:
        
        if count==0:
            plt.plot(taus, reads[ic][IT,:], color=palette[ic], label='students')
            count +=1
        else:
            plt.plot(taus, reads[ic][IT,:], color=palette[ic])
            
plt.plot(taus, read_true[IT,:], c='k', lw=1.5, alpha=0.8)
plt.legend(loc=1, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('readout', color='w')
ax.set_ylabel('')
ll = ax.get_ylim()
ax.set_xticklabels('')
plt.ylim([-1.2, 1.2])
plt.xlim([0, 16])

ax = fig.add_subplot(212)
IT =TRs[0][0]
for ic in range(ics):
    if valid[ic]:
        plt.plot(taus, reads[ic][IT,:], color=palette[ic])
plt.plot(taus, read_true[IT,:], c='k', lw=1.5, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time')
ax.set_ylabel('readout', color='w')

ll = ax.get_ylim()
IT =TRs[0][0]
plt.ylim([-1.2, 1.2])
plt.xlim([0, 16])

plt.savefig('Figs/F0_readoutBoth.pdf', transparent=True)
#%%
fig = plt.figure(figsize=[1.5*2.2*0.7, 1.5*2])
ax = fig.add_subplot(111)

delay = (np.array(inpTR)[0][0]-np.array(inpTR)[1][0])*dta
delay0 = np.array(inpTR)[0][0]*dta
plt.plot(taus-delay0, read_true[TRs[0][0],:],  c='k',lw=1.5, label='input 1')
plt.plot(taus+delay-delay0, read_true[TRs[1][0],:],  c=0.5*np.ones(3), alpha=0.5, lw=2.5, label='input 2')
#plt.legend(loc=4, fontsize=11)
plt.xlabel('time')
plt.ylim([-1.4, 1.4])
plt.xlim([-1, 13])
plt.ylabel('readout')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks([-1, 0, 1])
plt.savefig('Figs/F0_B.pdf', transparent=True)
plt.show()

#%%
fig = plt.figure(figsize=[1.5*2.2*0.7, 1.5*2])
ax = fig.add_subplot(111)

delay = (np.array(inpTR)[0][0]-np.array(inpTR)[1][0])*dta
delay0 = np.array(inpTR)[0][0]*dta
plt.plot(taus-delay0, input_train[TRs[0],:,1].detach().numpy().T,  c='k',lw=2, label='input 1')
#plt.plot(taus+delay-delay0, read_true[TRs[1][0],:],  c=0.5*np.ones(3), alpha=0.5, lw=2.5, label='input 2')
#plt.legend(loc=4, fontsize=11)
plt.xlabel('time')
#plt.ylim([-4, 4])
plt.xlim([-1, 13])
plt.ylim([-0.5, 1])
plt.ylabel('input timecourse')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks([ -0.5, 0, 0.5, 1])
ax.set_yticklabels(['','', '', ''])
plt.savefig('Figs/F0_B_2.pdf', transparent=True)
plt.show()

