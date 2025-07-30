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
stri = 'Data/Figure4/' 

seed =21
np.random.seed(seed)

#generate trials
input_train, output_train, cond_train = fl.create_input(taus, trials)



label = ''#''
Noise =0.5#0.1, 0.5
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


save_neurs = 100

count = 0
used_percs = percs#[0:16]
e=False
noises = np.array((0.005, 0.01,  0.02, 0.05, 0.1))
try:
    aa = np.load(stri+'data_plots_ef.npz')
    losses = aa['arr_0']
    losses_unk = aa['arr_1']
    losses_kno = aa['arr_2']
    losses_G = aa['arr_3']
    losses_B = aa['arr_4']
    #meG = aa['arr_5']
    #sameG = aa['arr_6']
except:
    if label=='':
        lr=0.002
    else:
        lr = 0.005
    for ino, noise in enumerate(noises):
        for ic in range(ics):
            for ip in range(len(percs)):
                try:
                    AA = np.load(stri+'task'+label+'_lossesMiCo_Dec'+str(noise)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
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
                    print(stri+'task'+label+'_lossesMiCo_Dec'+str(noise)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
                    print('ic '+str(ic))
                    print('nRec '+str(used_percs[ip]))
                    print('noise: '+str(noise))
                    print('error --')
                    e=True
        
                
                if count==0:
                    losses = np.zeros((len(losses_), len(used_percs), ics, len(noises) ))
                    losses_unk = np.zeros((len(losses_unk_), len(used_percs), (ics) , len(noises)))
                    losses_kno = np.zeros((len(losses_kno_), len(used_percs), (ics) , len(noises)))
                    losses_G = np.zeros((len(losses_kno_), len(used_percs), (ics) , len(noises)))
                    losses_B = np.zeros((len(losses_kno_), len(used_percs), (ics) , len(noises)))
                    
                    meG = np.zeros((np.shape(e_g)[1], len(used_percs), (ics) , len(noises)))
                    sameG = np.zeros((save_neurs, np.shape(e_g)[1], len(used_percs), (ics), len(noises) ))
                    
                    #i_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                    #f_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                if e:
                    losses[:,ip, ic, ino] = np.nan
                    losses_unk[:,ip, ic, ino] = np.nan
                    losses_kno[:,ip, ic, ino] = np.nan
                    losses_G[:,ip, ic, ino] = np.nan
                    losses_B[:,ip, ic, ino] = np.nan
                    meG[:,ip, ic, ino] = np.nan
                    #i_eG[:,:,ip, ic, ilr] = np.nan
                    #f_eG[:,:,ip, ic, ilr] = np.nan
                    #sameG[:,ip, ic, ilr] = np.nan
                    
                else:
                    losses[:,ip, ic, ino] = losses_
                    losses_unk[:,ip, ic, ino] = losses_unk_
                    losses_kno[:,ip, ic, ino] = losses_kno_
                    losses_G[:,ip, ic, ino] = losses_G_
                    losses_B[:,ip, ic, ino] = losses_B_
                    meG[:,ip, ic, ino] = np.sqrt(np.mean(e_g**2,axis=(0)))
                    #i_eG[:,:,ip, ic, ilr] = e_g[:,:,0]
                    #f_eG[:,:,ip, ic, ilr] = e_g[:,:,-1]
        
                    sameG[0:save_neurs,:,ip, ic, ino] = e_g[0:save_neurs,:]
                e=False
                
            count+=1
np.savez(stri+'data_plots_ef.npz', losses, losses_unk, losses_kno, losses_G, losses_B)

#%%
losses[losses==0] = np.nan
losses_unk[losses_unk==0] = np.nan
losses_kno[losses_kno==0] = np.nan


#%%
fl.set_plot()


#%%
used_percs=percs[::2]
norm = False
Lo = losses_kno
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

fig = plt.figure(figsize=[3*3.5, 2.7])

for ib, bi in enumerate(noises):
    ax = fig.add_subplot(1, len(noises), ib+1)
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
    ax.text(1500, 100, r'$\sigma=$'+str(bi))
plt.savefig('Figs/F6_noises_singleneuron_recloss.pdf', transparent=True)

#%%
norm = False
Lo = losses_kno
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)

paletteN = plt.cm.copper(np.linspace(0.2, 0.8, len(noises)))
for ib, bi in enumerate(noises):
    for ip, pi in enumerate(percs):
        
        #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
        # else:
        dat = Lo[-1,ip,:, ib]
        plt.scatter(pi*np.ones(len(dat)), dat, color=paletteN[ib], alpha=0.5, rasterized=True)
        plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=paletteN[ib], zorder=4, rasterized=True)
        if ip<len(percs)-1:
            plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:,ib])], c=paletteN[ib], lw=2)
        
        # if ip==0:
        #     dat = losses_kno[0,ip,:,ib]
        #     pi =0.5
        #     plt.scatter(pi*np.ones(len(dat)), dat, color=paletteN[ib], alpha=0.5, rasterized=True)
        #     plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=paletteN[ib], zorder=4, rasterized=True)

        #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
        #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
        # xx = np.arange(len(mm))*10
        # plt.plot(xx, mm, c=palette[ip])
        # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
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
ax.set_ylim([0.0005, 999])
ax.set_yticks([0.01, 1,  100])
ax.set_xscale('log')
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])


plt.savefig('Figs/F6_noiseall_singleneuron_rec_final.pdf', transparent=True, dpi = 300)
#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5*3, 2.7])
for ib, bi in enumerate(noises):
    ax = fig.add_subplot(1, len(noises), ib+1)

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
plt.savefig('Figs/F6_noise_singleneuron_unrecloss.pdf', transparent=True)




#%%
norm = False
Lo = losses_unk
palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
fig = plt.figure(figsize=[3.5, 2.7])
ax = fig.add_subplot(111)
for ib, bi in enumerate(noises):
    for ip, pi in enumerate(percs):
    
    #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
    #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
    # else:
        dat = Lo[-1,ip,:,ib]
        plt.scatter(pi*np.ones(len(dat)), dat, color=paletteN[ib], alpha=0.5, rasterized=True)
        plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=paletteN[ib], zorder=4, rasterized=True)
        if ip<len(percs)-1:
            plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:,ib])], c=paletteN[ib], lw=2)
        
        if ip==0:
            dat = losses_kno[0,ip,:,ib]
            pi =0.5
            plt.scatter(pi*np.ones(len(dat)), dat, color=paletteN[ib], alpha=0.5, rasterized=True)
            plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=paletteN[ib], zorder=4, rasterized=True)

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

ax.set_ylim([0.0005, 999])
ax.set_yticks([0.01, 1,  100])
ax.set_xscale('log')
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1', '10', '100'])


plt.savefig('Figs/F6_noiseall_singleneuron_unrec_final.pdf', transparent=True)


# #%%
# norm = False
# Lo = losses_G
# palette = plt.cm.plasma(np.linspace(0, 0.8, len(used_percs)))
# fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(211)
# for ip, pi in enumerate(used_percs):
#     if norm:
#         mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#     else:
#         mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#     xx = np.arange(len(mm))*10
#     mm = np.nanmean(mm,-1)
#     ss = np.nanmean(ss,-1)
#     plt.plot(xx, mm, c=palette[ip])
#     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
# ax.set_ylabel('error gain')
# #ax.set_xlabel('epochs')
# ax.set_xticklabels([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_ylim([0, 1.2])
# ax.set_yticks([0, 1.0])
# ax.set_yticklabels(['0.0', '1.0'])
# plt.savefig('Figs/F6_beta__singleneuron_gains.pdf', transparent=True)

# norm = False
# Lo = losses_B
# ax = fig.add_subplot(212)
# for ip, pi in enumerate(used_percs):
#     if norm:
#         mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#     else:
#         mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#     xx = np.arange(len(mm))*10
#     mm = np.nanmean(mm,-1)
#     ss = np.nanmean(ss,-1)
#     plt.plot(xx, mm, c=palette[ip])
#     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
# ax.set_ylabel('error bias')
# ax.set_xlabel('epochs')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# #ax.set_ylim([0, 7])
# ax.set_ylim([0, 0.7])
# #ax.set_yscale('log')
# plt.savefig('Figs/F6_beta__singleneuron_biases.pdf', transparent=True)
# #%% Till here
# #%%
# used_percs=percs[::2]
# norm = False
# Lo = losses_kno
# palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
# fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(111)
# for ip, pi in enumerate(percs):
#     if np.min(np.abs(pi-used_percs))==0:
#         if norm:
#             mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#             ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#         else:
#             mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#             ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#         xx = np.arange(len(mm))*10
#         mm = np.nanmean(mm,-1)
#         ss = np.nanmean(ss,-1)
#         plt.plot(xx, mm, c=palette[ip])
#         plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
# ax.set_ylabel('error recorded act.')
# ax.set_xlabel('epochs')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_yscale('log')
# ax.set_ylim([0.001, 9e2])
# plt.savefig('Figs/F6c_singleneuron_recloss_Noise'+str(Noise)+'.pdf', transparent=True)

# #%%
# norm = False
# Lo = losses_kno
# palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
# fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(111)
# for ip, pi in enumerate(percs):
    
#     #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#     #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#     # else:
#     dat = np.nanmean(Lo[-1,ip,:],-1)
#     plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5, rasterized=True)
#     plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette[ip], zorder=4, rasterized=True)
#     if ip<len(percs)-1:
#         plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
#     #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#     #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#     # xx = np.arange(len(mm))*10
#     # plt.plot(xx, mm, c=palette[ip])
#     # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
# ax.set_ylabel('error recorded act.')
# ax.set_xlabel('rec. units M')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_yscale('log')
# ax.set_ylim([0.0011, 999])
# ax.set_yticks([0.1, 10,  1000])
# ax.set_xscale('log')
# ax.set_xticks([1, 10, 100])
# ax.set_xticklabels(['1', '10', '100'])

# plt.savefig('Figs/F6c_singleneuron_rec_final_Noise'+str(Noise)+'.pdf', transparent=True)
# #%%
# norm = False
# Lo = losses_unk
# palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
# fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(111)
# for ip, pi in enumerate(percs):
#     if np.min(np.abs(pi-used_percs))==0:
#         if norm:
#             mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#             ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#         else:
#             mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#             ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#         xx = np.arange(len(mm))*10
#         mm = np.nanmean(mm,-1)
#         ss = np.nanmean(ss,-1)
#         plt.plot(xx, mm, c=palette[ip])
#         plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2, rasterized=True)
# ax.set_ylabel('error unrecorded act.')
# ax.set_xlabel('epochs')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_yscale('log')
# ax.set_ylim([0.001, 9e2])

# plt.savefig('Figs/F6c_singleneuron_unrecloss_Noise'+str(Noise)+'.pdf', transparent=True)

# #%%
# norm = False
# Lo = losses_unk
# palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
# fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(111)
# for ip, pi in enumerate(percs):
    
#     #     mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#     #     ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#     # else:
#     dat = np.nanmean(Lo[-1,ip,:],-1)
#     plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5, rasterized=True)
#     plt.scatter(pi, np.nanmean(dat), s=60, edgecolor='k', color=palette[ip], zorder=4, rasterized=True)
#     if ip<len(percs)-1:
#         plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
    
#     #     mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#     #     ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#     # xx = np.arange(len(mm))*10
#     # plt.plot(xx, mm, c=palette[ip])
#     # plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
# ax.set_ylabel('error unrecorded act.')
# ax.set_xlabel('rec. units M')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_yscale('log')
# ax.set_ylim([0.0011, 999])
# ax.set_yticks([0.1, 10,  1000])
# ax.set_xscale('log')
# ax.set_xticks([1, 10, 100])
# ax.set_xticklabels(['1', '10', '100'])

# plt.savefig('Figs/F6c_singleneuron_unrec_final_Noise'+str(Noise)+'.pdf', transparent=True)


# #%%
# norm = False
# Lo = losses_G
# palette = plt.cm.plasma(np.linspace(0, 0.8, len(used_percs)))
# fig = plt.figure(figsize=[3.5, 2.7])
# ax = fig.add_subplot(211)
# for ip, pi in enumerate(used_percs):
#     if norm:
#         mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#     else:
#         mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#     xx = np.arange(len(mm))*10
#     mm = np.nanmean(mm,-1)
#     ss = np.nanmean(ss,-1)
#     plt.plot(xx, mm, c=palette[ip])
#     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
# ax.set_ylabel('error gain')
# #ax.set_xlabel('epochs')
# ax.set_xticklabels([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_ylim([0, 1.2])
# ax.set_yticks([0, 1.0])
# ax.set_yticklabels(['0.0', '1.0'])
# plt.savefig('Figs/F6c_singleneuron_gains_Noise'+str(Noise)+'.pdf', transparent=True)

# norm = False
# Lo = losses_B
# ax = fig.add_subplot(212)
# for ip, pi in enumerate(used_percs):
#     if norm:
#         mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
#     else:
#         mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
#         ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
#     xx = np.arange(len(mm))*10
#     mm = np.nanmean(mm,-1)
#     ss = np.nanmean(ss,-1)
#     plt.plot(xx, mm, c=palette[ip])
#     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
# ax.set_ylabel('error bias')
# ax.set_xlabel('epochs')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# #ax.set_ylim([0, 7])
# ax.set_ylim([0, 0.7])
# #ax.set_yscale('log')
# plt.savefig('Figs/F6c_singleneuron_biases_Noise'+str(Noise)+'.pdf', transparent=True)
# #%%
# dtype = torch.FloatTensor 
# input_train, output_train, cond_train = fl.create_input(taus, trials)
# input_train = torch.from_numpy(input_train).type(dtype)
# output_train = torch.from_numpy(output_train).type(dtype)
# #%%
# do_singleplots = True
# if do_singleplots:
#     # Plot error
#     ic  = 5 #6 is good
#     IPS = np.arange(len(used_percs))#[3, 5, 6, 7, 8, 12, 14]
#     for IP in IPS:
    
#         tri = 2
#         print(used_percs[IP])
#         AA = np.load(stri+'task'+label+'_lossesMiCo'+str(Noise)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(lr)+'_unpacked.npz')
        
#         e_g = AA['arr_5']
#         e_b = AA['arr_6']
#         JN = AA['arr_9']
        
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
#         wrec2= torch.from_numpy(JN).type(dtype)
        
#         H0 = torch.from_numpy(x0).type(dtype)
#         mwrec = torch.from_numpy(mwrec_).type(dtype)
#         refEI = torch.from_numpy(refEI_).type(dtype)
#         wo_init2 = np.eye(N)
#         wo_init2 = torch.from_numpy(wo_init2).type(dtype)
        
#         Net_True = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
#                                   train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
#         NetJ2    = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec2, mwrec, refEI, b_Tra, g_Tra, h0_init=H0,
#                                   train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
#         Out_True = Net_True(input_train)
#         out_true = Out_True.detach().numpy()
        
#         Out_Tra = NetJ2(input_train)
#         out_tra = Out_Tra.detach().numpy()
        
#         selecN = np.arange(300)
#         selecT = 3
#         vMa = 10.
#         fig = plt.figure(figsize=[3.5, 3.2])
#         ax = fig.add_subplot(111)
#         plt.imshow(np.abs(out_tra[tri,::selecT,selecN]-out_true[tri,::selecT,selecN]), cmap='Greys', vmin=0, vmax=vMa, origin='lower', rasterized=True)
#         rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor='k', alpha=0.5, facecolor='none')
#         ax.add_patch(rect)
#         rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor=palette[IP], alpha=0.9, facecolor='none')
#         ax.add_patch(rect)
#         ax.set_ylabel('neuron id')
#         ax.set_xlabel('time')
#         ax.set_xticks([0, 100])#labels('')
#         ax.set_xticklabels(['0', '15'])
#         plt.savefig('Figs/F6c_example_Nrec_'+str(used_percs[IP])+'_legend_Noise'+str(Noise)+'.pdf', dpi=300, transparent=True)
#         plt.show()
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
            
            
#             plt.savefig('Figs/F6c_example_legend_Noise'+str(Noise)+'.pdf', dpi=300, transparent=True)
#             plt.show()
    
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
#         plt.savefig('Figs/F6c_readout_Nrec_'+str(used_percs[IP])+'_legend_Noise'+str(Noise)+'.pdf', dpi=300, transparent=True)
#         plt.show()
        
#         fig = plt.figure(figsize=[1.5*2.2*0.8, 1.5*2])
#         ax = fig.add_subplot(111)
        
#         delay = (np.array(inpTR)[0][0]-np.array(inpTR)[1][0])*dta
#         plt.plot(taus, reads_true[1,:],  c='k',lw=1.5, label='input 1')
#         plt.plot(taus-delay, reads_true[0,:],  c=0.5*np.ones(3), alpha=0.5, lw=3.5, label='input 2')
#         #plt.legend(loc=4, fontsize=11)
#         plt.xlabel('time')
#         plt.ylim([-2, 2])
#         plt.xlim([3, 16])
#         plt.ylabel('readout')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         ax.set_yticks([-1, 0, 1])
#         plt.savefig('Figs/F6c_B.pdf', transparent=True)
#         plt.show()
        
        
    
#         fig = plt.figure(figsize=[1.5*2.2, 1.5*2*0.7])
#         ax = fig.add_subplot(111)
#         dat =e_g[0:used_percs[IP],-1]
        
#         sd = 0.05
        
#         val = 0.
#         plt.scatter(sd*np.random.randn(len(dat)), dat, edgecolor='k', linewidth=0.3, color='C5', rasterized=True)    
#         parts = ax.violinplot(
#             dat, showmeans=False, showmedians=False,
#             showextrema=False, positions=[val])
#         for pc in parts['bodies']:
#             pc.set_edgecolor('black')
#             pc.set_facecolor('C5')
#             pc.set_alpha(0.5)
#         plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#         plt.scatter([val], [np.mean(dat)], s=50, color='C5', edgecolor='k', zorder=3, rasterized=True)    
    
#         val = 0.4
#         dat =e_g[used_percs[IP]:,-1]
        
#         parts = ax.violinplot(
#             dat, showmeans=False, showmedians=False,
#             showextrema=False, positions=[val])
#         for pc in parts['bodies']:
#             pc.set_edgecolor('black')
#             pc.set_alpha(0.5)
#             pc.set_facecolor(np.ones(3)*0.3)
#         plt.scatter(val+sd*np.random.randn(len(dat)), dat, edgecolor=0.1*np.ones(3), color='w', linewidth=0.3, rasterized=True)  
#         plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#         plt.scatter([val], [np.mean(dat)], s=50, color='w', edgecolor='k', zorder=3, rasterized=True)  
            
        
#         val = 1.1
#         dat =e_b[0:used_percs[IP],-1]
    
#         parts = ax.violinplot(
#             dat, showmeans=False, showmedians=False,
#             showextrema=False, positions=[val])
#         for pc in parts['bodies']:
#             pc.set_edgecolor('black')
#             pc.set_alpha(0.5)
#             pc.set_facecolor('C6')
#         plt.scatter(val+sd*np.random.randn(len(dat)), dat, edgecolor='k', color='C6', linewidth=0.3, rasterized=True)    
#         plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#         plt.scatter([val], [np.mean(dat)], s=50, color='C6', edgecolor='k', zorder=3, rasterized=True)          
        
#         val = 1.5
#         dat =e_b[used_percs[IP]:,-1]
     
#         parts = ax.violinplot(
#             dat, showmeans=False, showmedians=False,
#             showextrema=False, positions=[val])
#         for pc in parts['bodies']:
#             pc.set_edgecolor('black')
#             pc.set_alpha(0.5)
#         plt.scatter(val+sd*np.random.randn(len(dat)), dat, edgecolor='k', color='w', linewidth=0.3, rasterized=True)   
#         plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#         plt.scatter([val], [np.mean(dat)], s=50, color='w', edgecolor='k', zorder=3, rasterized=True) 
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')    
#         ax.set_xticks([0, 0.4, 1.1, 1.5])
#         if used_percs[IP]==30:
#             ax.set_xticklabels([r'$g_{rec}$', r'$g_{unrec}$', r'$b_{rec}$', r'$b_{unrec}$',], fontsize=14, rotation=30)
#         else:
#             ax.set_xticklabels([r'$g_{rec}$', r'$g_{unrec}$', r'$b_{rec}$', r'$b_{unrec}$',], fontsize=14, rotation=30, color='w')
#         ax.set_ylabel('error')
#         ax.set_yticks([-1, 0, 1])
        
#         plt.savefig('Figs/F6c_errorHist_Nrec_'+str(used_percs[IP])+'_legend_Noise'+str(Noise)+'.pdf', dpi=300, transparent=True)
#         plt.show()
        
    

# #%%
# if do_singleplots:
#     fig = plt.figure(figsize=[1.5*2.2*1., 1.5*2*1.])
#     ax = fig.add_subplot(111)
#     ax.imshow(J, vmin = -0.2, vmax=0.2, cmap='bwr', origin='lower')
#     ax.set_xlabel('pre-syn neuron')
#     ax.set_ylabel('post-syn neuron')
#     ax.set_yticks([0, 100, 200, 300])
#     ax.set_xticks([0, 100, 200, 300])
#     plt.savefig('Figs/F6c_B_conn.pdf', transparent=True)
    
#     #%%
# fig = plt.figure()  
# ax = fig.add_subplot(111)
# J = Net_True.wrec.detach().numpy()
# uJ = np.linalg.eigvals(J)
# ICs = 10
# uJN = []

# np.random.seed(10)
# f_net = np.load(stri+'teacherRNN_cycling.npz')
# #f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
# J = f_net['arr_0']

# Chi2 = np.zeros_like(J)
# paletteN = plt.cm.copper(np.linspace(0.2, 0.8, len(noises)))

# for inn, noi in enumerate(noises):
#     p_skconn = noi
#     sig_J= p_skconn
    
#     for j in range(N):
#         for i in range(N):
#             if J[i,j]==0:
#                 if np.random.rand()>p_skconn:
#                     continue
#                 else:
#                     if refEI[0,j]>0:
#                         Chi2[i,j] = np.random.permutation(J[J[:,j]>0,j])[0]
#                     else:
#                         Chi2[i,j] = np.random.permutation(J[J[:,j]<0,j])[0]
#             else:
#                 Chi2[i,j] = J[i,j]*(1+np.random.randn()*sig_J)
                
#     uJN_ = np.linalg.eigvals(Chi2)
#     plt.scatter(np.real(uJN_), np.imag(uJN_), c=paletteN[inn], rasterized=True)
#     uJN.append(uJN_)
# #plt.scatter(np.real(uJN_), np.imag(uJN_), c=0.5*np.ones(3), label='students', rasterized=True)
# plt.scatter(np.real(uJ), np.imag(uJ), c=0.5*np.ones(3), s=20, label='teacher', rasterized=True, edgecolor='k', alpha=0.7)

# #plt.legend(frameon=False, handletextpad=0.1)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom') 
# ax.set_xlabel(r'Re($\lambda_J$)')
# ax.set_ylabel(r'Im($\lambda_J$)')

# plt.savefig('Figs/F6c_eigvals.pdf', transparent=True)

#     #%%
# fig = plt.figure()  
# ax = fig.add_subplot(111)
# J = Net_True.wrec.detach().numpy()
# uJ = np.linalg.eigvals(J)
# ICs = 10
# uJN = []

# np.random.seed(10)
# #f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
# J = f_net['arr_0']
# noises = np.array((0.005, 0.01, 0.02, 0.05, 0.1))
# Chi2 = np.zeros_like(J)
# paletteN = plt.cm.copper(np.linspace(0.2, 0.8, len(noises)))
# plt.scatter(np.real(uJ), np.imag(uJ), c=0.5*np.ones(3), s=20, label='teacher', rasterized=True, edgecolor='k', alpha=0.7)
# for inn, noi in enumerate(noises):
#     p_skconn = noi
#     sig_J= p_skconn
    
#     for j in range(N):
#         for i in range(N):
#             if J[i,j]==0:
#                 if np.random.rand()>p_skconn:
#                     continue
#                 else:
#                     if refEI[0,j]>0:
#                         Chi2[i,j] = np.random.permutation(J[J[:,j]>0,j])[0]
#                     else:
#                         Chi2[i,j] = np.random.permutation(J[J[:,j]<0,j])[0]
#             else:
#                 Chi2[i,j] = J[i,j]*(1+np.random.randn()*sig_J)
                
#     uJN_ = np.linalg.eigvals(Chi2)
#     plt.scatter(np.real(uJN_), np.imag(uJN_), c=paletteN[inn], rasterized=True, label=r'$\sigma=$'+str(noi))
#     uJN.append(uJN_)
# #plt.scatter(np.real(uJN_), np.imag(uJN_), c=0.5*np.ones(3), label='students', rasterized=True)


# plt.legend(frameon=False, handletextpad=0.1)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom') 
# ax.set_xlabel(r'Re($\lambda_J$)')
# ax.set_ylabel(r'Im($\lambda_J$)')
# ax.set_xlim([-20, -21])

# plt.savefig('Figs/F6c_eigvals_2.pdf', transparent=True)
# #%%

# # #%%
# # Vmin = 0
# # Vmax = 15
# # fig = plt.figure(figsize=[3, 2])
# # ax = fig.add_subplot(111)
# # bl = plt.pcolor(taus, np.arange(N), ((out_tra[0,:,:]).T), vmin=Vmin, vmax=Vmax, cmap = 'Greys', shading='auto', rasterized=True)
# # fig.colorbar(bl, ticks = [ 0, ])
# # ax.set_ylabel('neuron index')
# # ax.set_xlabel('time')
# # ax.set_xticks([0, 15])
# # ax.set_xticklabels(['0', '15'])
# # ax.set_yticks([0, 120, 300])
# # ax.set_yticklabels(['1', 'M', 'N'])
# # ax.set_xticklabels(['0', 'T'])
# # plt.savefig('Figs/F6c_A_act.pdf', dpi=300, transparent=True)

# # #%%
# # x = np.linspace(-3.5, 3.5)
# # y = fl.softpl(x)

# # x2 = np.copy(x)
# # y2 = 2*fl.softpl(x2)
# # fig = plt.figure(figsize=[1.5*2.2*0.7, 1.5*2])
# # ax = fig.add_subplot(111)
# # #plt.plot(x,y, c='k', lw=3, zorder=3)

# # for shift in np.arange(-2, 2, 0.5):
# #     x1 = np.linspace(-3.5, 3.5-shift)
# #     y1 = fl.softpl(x1+shift)
# #     plt.plot(x1,y1, c=0.5*np.ones(3), alpha=0.8)
# # #plt.plot(x2,y2)
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.yaxis.set_ticks_position('left')
# # ax.xaxis.set_ticks_position('bottom')  
# # plt.xlabel('input')
# # plt.ylabel('output activity', color='w')
# # plt.savefig('Figs/F6c_B_bias.pdf', dpi=300, transparent=True)

# # #%%
# # x = np.linspace(-3.5, 3.5)
# # y = fl.softpl(x)


# # fig = plt.figure(figsize=[1.5*2.2*0.7, 1.5*2])
# # ax = fig.add_subplot(111)
# # #plt.plot(x,y, c='k', lw=3, zorder=3)

# # for shift in np.arange(0.4, 1.5, 0.15):
# #     x1 = np.copy(x)
# #     y1 = shift*fl.softpl(x1)
# #     plt.plot(x1,y1, c=0.5*np.ones(3), alpha=0.8)
# # #plt.plot(x2,y2)
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.yaxis.set_ticks_position('left')
# # ax.xaxis.set_ticks_position('bottom')  
# # plt.xlabel('input')
# # plt.ylabel('output activity')
# # plt.savefig('Figs/F6c_B_gains.pdf', dpi=300, transparent=True)

# #%%
# # #%%

# # Co=0
# # Dat = out_tra
# # for it in range(np.shape(Dat)[0]):
# #     X = Dat[it,:,:]-np.mean(np.mean(Dat,0),0)
# #     Co  += X.T.dot(X)/np.shape(Dat)[0]    
# # eva, evs = np.linalg.eig(Co)

# # Proj = np.zeros_like(Dat)
# # for it in range(np.shape(Dat)[0]):
# #     Proj[it,:,:] = np.dot( Dat[it,:,:]-np.mean(np.mean(Dat,0),0), evs)
# # #%%
# # PR = (np.sum(eva)**2)/np.sum(eva**2)
# # print(PR)
# # #%%
# # plt.plot(np.cumsum(eva)/np.sum(eva))
# # #%%
# # iT = 20#
# # plt.plot(Proj[iT,:,0])
# # #plt.plot(Proj[iT,:,1])
# # #plt.plot(Proj[iT,:,2])
# # #plt.plot(Proj[iT,:,3])
# # #plt.plot(Proj[iT,:,4])
# # plt.plot(Proj[iT,:,24])

