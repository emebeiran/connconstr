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


#%%
f_net = np.load(stri+'teacherRNN_cycling.npz')
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

palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
ics = 10

losses_ = []
eG_ = []

losses = []
eG = []


save_neurs = 100

count = 0
percs = percs#[0:13]
used_percs = percs#[0:16]
e=False
lr=0.001
try:
    #np.savez(stri+'losses_fig2_bdf.npz', losses, losses_unk, losses_kno, 
        # losses_J, losses_J_unk, losses_J_kno, f_wrec)
    bb = np.load(stri+'losses_fig2_bdf.npz')
    losses = bb['arr_0']
    losses_unk = bb['arr_1']
    losses_kno = bb['arr_2']
    losses_J = bb['arr_3']
    losses_J_unk = bb['arr_4']
    losses_J_kno = bb['arr_5']
    #f_wrec = bb['arr_6']
except:
    print('loading datafiles for losses')
    for ic in range(ics):
        for ip in range(len(percs)-1):
            #AA = np.load(stri+'task_losses_percSGD_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_unpacked.npz')
            try:
                #losses, losses_unk, losses_kno, losses_J, losses_J_unk, losses_J_kno, wF, mask_l, losses_unkWr
                AA = np.load(stri+'taskConnDecNoEI_losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                losses_ = AA['arr_0']
                #losses_read_ = AA['arr_1']
                losses_unk_ = AA['arr_1']
                losses_kno_ = AA['arr_2']
                losses_J_ = AA['arr_3']
                losses_J_unk_ = AA['arr_4']
                losses_J_kno_ = AA['arr_5']
                wF_ = AA['arr_6']
                mask_l = AA['arr_7']
                #losses_unkWr_ = AA['arr7']
            
                #losses[::10],  losses_unk, losses_kno, losses_J, losses_J_unk, losses_J_kno, wF, losses_unkWr
                if len(losses_)>700:
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
                losses_J = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                losses_J_kno = np.zeros((len(losses_J_kno_), len(used_percs), (ics) ))
                losses_J_unk = np.zeros((len(losses_J_unk_), len(used_percs), (ics) ))
                #losses_unkWr = np.zeros((len(losses_unkWr_), len(used_percs), (ics) ))
                count+=1
                
                
                #meG = np.zeros((np.shape(e_g)[1], len(used_percs), (ics) ))
                #sameG = np.zeros((save_neurs, np.shape(e_g)[1], len(used_percs), (ics) ))
                
                #i_w = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                #f_wrec = np.zeros((np.shape(wF_)[0], np.shape(wF_)[0], len(used_percs), (ics) ))
            if e:
                losses[:,ip, ic] = np.nan
                losses_unk[:,ip, ic] = np.nan
                losses_kno[:,ip, ic] = np.nan
                losses_J[:,ip, ic] = np.nan
                losses_J_unk[:,ip, ic] = np.nan
                losses_J_kno[:,ip, ic] = np.nan
                #losses_unkWr[:,ip, ic] = np.nan
                #f_wrec[:,:,ip, ic] = np.nan
                
                #meG[:,ip, ic] = np.nan
                #i_eG[:,:,ip, ic, ilr] = np.nan
                #f_eG[:,:,ip, ic, ilr] = np.nan
                #sameG[:,ip, ic, ilr] = np.nan
                
            else:
                losses[:,ip, ic] = losses_
                losses_unk[:,ip, ic] = losses_unk_
                losses_kno[:,ip, ic] = losses_kno_
                losses_J[:,ip, ic] = losses_J_
                losses_J_unk[:,ip, ic] = losses_J_unk_
                losses_J_kno[:,ip, ic] = losses_J_kno_
                #losses_unkWr[:,ip, ic] = losses_unkWr_
                #f_wrec[:,:,ip, ic] = wF_

                    
                #meG[:,ip, ic] = np.sqrt(np.mean(e_g**2,axis=(0)))
                #i_eG[:,:,ip, ic, ilr] = e_g[:,:,0]
                #f_eG[:,:,ip, ic, ilr] = e_g[:,:,-1]

                #sameG[0:save_neurs,:,ip, ic] = e_g[0:save_neurs,:]
            e=False
            
        count+=1
        #eG= AA['arr_4']
    np.savez(stri+'losses_fig2_bdf.npz', losses, losses_unk, losses_kno, 
            losses_J, losses_J_unk, losses_J_kno)


#%%
dtype = torch.FloatTensor 
input_train, output_train, cond_train = fl.create_input(taus, trials)
input_train = torch.from_numpy(input_train).type(dtype)
output_train = torch.from_numpy(output_train).type(dtype)
#%%
# Plot error
try:
    bbb = np.load(stri+'err_output.npz')
    E_J = bbb['arr_1']
    E_act = bbb['arr_0']
except:
    ic  = 0 #6 is good
    IPS = np.arange(len(used_percs))#[3, 5, 6, 7, 8, 12, 14]
    for IP in IPS:
        tri = 2
        print(used_percs[IP])
        AA = np.load(stri+'taskConnDecNoEI_losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
        wTra = AA['arr_6']
        mask_l = AA['arr_7']
    
        
        #f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
        J = f_net['arr_0']
        gt = f_net['arr_1']
        bt = f_net['arr_2']
        wI = f_net['arr_3']
        wout = f_net['arr_4']
        x0 = f_net['arr_5']
        mwrec_ = f_net['arr_6']
        mwrec2_ = np.ones_like(mwrec_)
        refEI_ = f_net['arr_7']
        refVec = refEI[0,:]
        
    
        Mask_l = np.arange(N)
        Mask_l[used_percs[IP]:] = mask_l+used_percs[IP]
    
        
        input_size = np.shape(wI)[0]
        N = np.shape(wI)[1]
        hidden_size = N
        
        g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
        #g_Tra  = torch.from_numpy(gtr[:,np.newaxis]).type(dtype)
        b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
        #b_Tra  = torch.from_numpy(btr[:,np.newaxis]).type(dtype)
        
        wi_init = torch.from_numpy(wI).type(dtype)
        wrec= torch.from_numpy(J).type(dtype)
        wrec_Tra = torch.from_numpy(wTra).type(dtype)
        
        H0 = torch.from_numpy(x0).type(dtype)
        mwrec = torch.from_numpy(mwrec_).type(dtype)
        mwrec2 = torch.from_numpy(mwrec2_).type(dtype)
        
        refEI = torch.from_numpy(refEI_).type(dtype)
        refEI2_ = np.copy(refEI)
        refEI2_[:,used_percs[IP]:]= 0.
        refEI2 = torch.from_numpy(0*refEI2_).type(dtype)
        wo_init2 = np.eye(N)
        wo_init2 = torch.from_numpy(wo_init2).type(dtype)
        
        Net_True = fl.RNNgain(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                                  train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
        NetJ2    = fl.RNNgain(input_size, hidden_size, N, wi_init, wo_init2, wrec_Tra, mwrec2, refEI2, b_True, g_True, h0_init=H0,
                                  train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
    
    
        Out_True = Net_True(input_train)
        out_true = Out_True.detach().numpy()
        
        Out_Tra = NetJ2(input_train)
        out_tra = Out_Tra.detach().numpy()
    
        out_tra2 = out_tra[:,:,Mask_l]
        
        selecN = np.arange(300)
        selecT = 3
        vMa = 10.
        fig = plt.figure(figsize=[3.5, 3.2])
        ax = fig.add_subplot(111)
        plt.imshow(np.abs(out_tra[tri,::selecT,selecN]-out_true[tri,::selecT,selecN]), cmap='Greys', vmin=0, vmax=vMa, origin='lower', rasterized=True) #'Reds'
        rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor='k', alpha=0.5, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor=palette[IP], alpha=0.9, facecolor='none')
        ax.add_patch(rect)
        ax.set_ylabel('neuron id')
        ax.set_xlabel('time')
        ax.set_xticks([0, 100])#labels('')
        ax.set_xticklabels(['0', '15'])
        plt.savefig('Figs/F1Connexample_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
        plt.show()
        
        fig = plt.figure(figsize=[3.5, 3.2])
        ax = fig.add_subplot(111)
        plt.imshow(np.abs(out_tra2[tri,::selecT,selecN]-out_true[tri,::selecT,selecN]), cmap='Greys', vmin=0, vmax=vMa, origin='lower', rasterized=True)
        rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor='k', alpha=0.5, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor=palette[IP], alpha=0.9, facecolor='none')
        ax.add_patch(rect)
        ax.set_ylabel('neuron id')
        ax.set_xlabel('time')
        ax.set_xticks([0, 100])#labels('')
        ax.set_xticklabels(['0', '15'])
        plt.savefig('Figs/F1Connexample_Nrec_'+str(used_percs[IP])+'_legend_rearrange.pdf', dpi=300, transparent=True)
        plt.show()
        
        print('losses')
        print(np.mean(np.abs(out_tra2[tri,::selecT,selecN]-out_true[tri,::selecT,selecN])))
        print(np.mean(np.abs(out_tra[tri,::selecT,selecN]-out_true[tri,::selecT,selecN])))
        print('--') 
        if IP==IPS[-1]:
            fig = plt.figure(figsize=[3.5, 3.2])
            ax = fig.add_subplot(111)
            plt.imshow(np.abs(out_tra[tri,::selecT,selecN]-out_true[tri,::selecT,selecN]), cmap='Reds', vmin=0, vmax=vMa, rasterized=True, origin='lower')
            cb = plt.colorbar()
            rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor='k', alpha=0.5, facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor=palette[IP], alpha=0.9, facecolor='none')
            ax.add_patch(rect)
            ax.set_ylabel('neuron id')
            ax.set_xlabel('time')
            ax.set_xticks([0, 100])#labels('')
            ax.set_xticklabels(['0', '15'])
            
            plt.savefig('Figs/F1Connexample_legend.pdf', dpi=300, transparent=True)
            plt.show()
    
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
        reads_true = np.zeros((len(TRs), np.shape(out_true)[1]))
        reads_tra = np.zeros((len(TRs), np.shape(out_true)[1]))
        reads_tra2 = np.zeros((len(TRs), np.shape(out_true)[1]))
        
        
        for itR, tR in enumerate(TRs):
            reads_true[itR,:] = np.dot(wout, out_true[tR[0],:,:].T)
            reads_tra[itR,:] = np.dot(wout, out_tra[tR[0],:,:].T)
            reads_tra2[itR,:] = np.dot(wout, out_tra2[tR[0],:,:].T)
            
    
        fig = plt.figure(figsize=[1.5*2.2, 1.5*2*0.7])
        ax = fig.add_subplot(111)
        plt.plot(taus, reads_true[0,:],  '--', c='k',lw=0.8)
        #plt.plot(taus, reads_tra[0,:], '--', lw=1., color=palette[IP])
        plt.plot(taus, reads_tra2[0,:], lw=1.5, color=palette[IP])
        
        plt.xlabel('time')
        plt.ylabel('readout')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks([-1, 0, 1])
        plt.savefig('Figs/F1Connreadout_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
        plt.show()

    pers = 50
    e_act_ = np.zeros(pers)
    e_J_ = np.zeros(pers)
    
    for ipe in range(pers):
        e_act_[ipe] = np.mean((out_true-out_true[:,:,np.random.permutation(N)])**2)
        xpe=  np.random.permutation(N)
        e_J_[ipe] = np.mean((J.dot(np.diag(gt))-J[np.ix_(xpe, xpe)].dot(np.diag(gt)))**2)
    E_act =np.mean(e_act_)
    E_J = np.mean(e_J_)
    np.savez(stri+'err_output.npz', E_act, E_J)

#%%
fl.set_plot()


#%%
used_percs=percs[::2]
norm = False
Lo = losses_kno#losses_kno
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
ax.set_ylabel('error recorded act.')
ax.set_xlabel('epochs')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
ax.set_yticks([0.001, 0.1, 10])
ax.set_ylim([0.0001, 50])
plt.savefig('Figs/F1Conn_singleneuron_recloss.pdf', transparent=True)

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
    plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5, rasterized=True)
    plt.scatter(pi, np.nanmean(dat), s=30, edgecolor='k', color=palette[ip], zorder=4, rasterized=True)
    if ip<len(percs)-2:
        plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
#ax.set_ylabel('error recorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
#ax.set_yticks([0.001, 0.1, 10])
ax.set_yticks([0.001, 0.1, 10])
#ax.set_yticklabels([1e-3, '0.1', '10'])
ax.set_xticks([1, 10, 100])
ax.set_ylabel('error recorded act.')
ax.set_xticklabels(['1', '10', '100'])
ax.set_ylim([0.0001, 50])
ax.set_xscale('log')
plt.savefig('Figs/F1Connsingleneuron_rec_final.pdf', transparent=True)
#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ip, pi in enumerate(percs):
    if np.min(np.abs(pi-used_percs))==0:
        mask =np.isnan(np.sum(np.sum(Lo,0),0))
        if norm:
            mm = np.nanmean((Lo[:,ip,~mask]/Lo[0,ip,~mask]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,~mask]/Lo[0,ip,:]), -1)/np.sqrt(np.sum(~mask))#/Lo[0,ip,:]
        else:
            mm = np.nanmean((Lo[:,ip,~mask]),-1)#/np.nanmean(Lo[0,ip,:], -1)
            ss = np.nanstd((Lo[:,ip,~mask]), -1)/np.sqrt(np.sum(~mask))#/Lo[0,ip,:]        
        xx = np.arange(len(mm))*10
        plt.plot(xx, mm, c=palette[ip])
        if ip==0:
            MM = mm
            Mask = mask
            #print(mm)
        plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('epochs')
xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*E_act, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
ax.set_yticks([0.1, 1, 10])
#ax.set_yticklabels([0.1, 1, 10])
ax.set_ylim([0.01, 50])
plt.savefig('Figs/F1Connsingleneuron_unrecloss.pdf', transparent=True)

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
    plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5)
    plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette[ip], zorder=4)
    if ip<len(percs)-2:
        plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
    #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
    # xx = np.arange(len(mm))*10
    # plt.plot(xx, mm, c=palette[ip])
    # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
    
pi = 0.5
dat = Lo[0,2,:]
plt.scatter(pi*np.ones(len(dat)), dat, color='k', alpha=0.5, rasterized=True)
plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color='k', zorder=4, rasterized=True)
ax.set_xscale('log')
xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*E_act, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
ax.set_ylabel('error unrecorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')

ax.set_yticks([0.1, 1, 10])
#ax.set_yticklabels([0.1, 1, 10])
ax.set_ylim([0.01, 50])
#ax.set_xlim([1, 300])
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])
plt.savefig('Figs/F1Connsingleneuron_unrec_final.pdf', transparent=True)


#%%
norm = False
Lo = losses_J
palette = plt.cm.plasma(np.linspace(0, 0.8, len(used_percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
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
ax.set_ylabel('error conn. J')
#ax.set_xlabel('epochs')
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_yscale('log')
plt.savefig('Figs/F1Connsingleneuron_J.pdf', transparent=True)
#%%
norm = False
Lo = losses_J_kno
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
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
ax.set_ylabel('error conn. J')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*E_J, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl) 
#plt.yscale('log')  
plt.ylim([0., 0.08])
plt.savefig('Figs/F1Connsingleneuron_Jkno.pdf', transparent=True)

#%%
norm = False
Lo = losses_J_unk
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
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
ax.set_ylabel('error conn. J')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# if not norm:
#     ax.set_ylim([0, 0.04])
#     ax.set_yticks([0, 0.02, 0.04])
#     ax.set_yticklabels([0, 0.02, 0.04])
xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*E_J, lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
yl = ax.get_ylim()
ax.set_ylim([0, yl[1]])
#ax.set_ylim([0, 7])
#ax.set_ylim([0, 0.7])
plt.ylim([0., 0.08])

    
plt.savefig('Figs/F1Connsingleneuron_Junkno.pdf', transparent=True)



