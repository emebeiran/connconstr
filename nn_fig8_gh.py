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
stri = 'Data/Figure8/' 

seed =21
np.random.seed(seed)

#generate trials
input_train, output_train, cond_train = fl.create_input(taus, trials)


#%%

#f_net = np.load('example_net_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
f_net = np.load('Data/Figure1/teacherRNN_cycling.npz')
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


used_percs = percs#[0:16]

labels = ['random_', 'best_', 'worst_']#['random_', 'bestmagic_', 'worstmagic_']
labels_all = [labels[0], labels[1], labels[2], 'random1_', 'random2_', 'random3_']

try:
    bb = np.load(stri+'data_fig8_gh.npz')
    Losses = bb['arr_0']
    Losses_unk = bb['arr_1']
    Losses_kno = bb['arr_2']
    Losses_G = bb['arr_3']
    Losses_B = bb['arr_4']
except:
    print('loading losses')
    Losses = []
    Losses_unk = []
    Losses_kno = []
    Losses_G = []
    Losses_B = []
    for il, la in enumerate(labels_all):
        count = 0
        e=False
        if la=='bestmagic_' or la=='best_':
            lr= 0.001
        elif la=='worstmagic_' or la=='worst__':
            lr = 0.005
        else:
            lr = 0.005 #0.01 for ''
        for ic in range(ics):
            for ip in range(len(percs)):
                #AA = np.load(stri+'task_losses_percSGD_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_unpacked.npz')
                try:
                    AA = np.load(stri+'task_'+la+ 'losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[ip])+'_lr_'+str(lr)+'_unpacked.npz')
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
                    adsfd
                    print('ic '+str(ic)+'nRec '+str(used_percs[ip])+' error --'+la)
                    e=True
        
                
                if count==0:
                    losses = np.zeros((len(losses_), len(used_percs), ics ))
                    losses_unk = np.zeros((len(losses_unk_), len(used_percs), (ics) ))
                    losses_kno = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                    losses_G = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                    losses_B = np.zeros((len(losses_kno_), len(used_percs), (ics) ))
                    
                    meG = np.zeros((np.shape(e_g)[1], len(used_percs), (ics) ))
                    sameG = np.zeros((save_neurs, np.shape(e_g)[1], len(used_percs), (ics) ))
                    
                    #i_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                    #f_eG = np.zeros((np.shape(e_g)[1], np.shape(e_g)[1], len(used_percs), (ics) ))
                if e:
                    losses[:,ip, ic] = np.nan
                    losses_unk[:,ip, ic] = np.nan
                    losses_kno[:,ip, ic] = np.nan
                    losses_G[:,ip, ic] = np.nan
                    losses_B[:,ip, ic] = np.nan
                    meG[:,ip, ic] = np.nan
                    #i_eG[:,:,ip, ic, ilr] = np.nan
                    #f_eG[:,:,ip, ic, ilr] = np.nan
                    #sameG[:,ip, ic, ilr] = np.nan
                    
                else:
                    losses[:,ip, ic] = losses_
                    losses_unk[:,ip, ic] = losses_unk_
                    losses_kno[:,ip, ic] = losses_kno_
                    losses_G[:,ip, ic] = losses_G_
                    losses_B[:,ip, ic] = losses_B_
                    meG[:,ip, ic] = np.sqrt(np.mean(e_g**2,axis=(0)))
                    #i_eG[:,:,ip, ic, ilr] = e_g[:,:,0]
                    #f_eG[:,:,ip, ic, ilr] = e_g[:,:,-1]
        
                    sameG[0:save_neurs,:,ip, ic] = e_g[0:save_neurs,:]
                e=False
                
            count+=1
        Losses.append(losses)
        Losses_unk.append(losses_unk)
        Losses_kno.append(losses_kno)
        Losses_G.append(losses_G)
        Losses_B.append(losses_B)
        
    Losses = np.array(Losses)
    Losses_unk = np.array(Losses_unk)
    Losses_kno = np.array(Losses_kno)
    Losses_G = np.array(Losses_G)
    Losses_B = np.array(Losses_B)
    np.savez(stri+'data_fig8_gh.npz', Losses, Losses_unk, Losses_kno, Losses_G, Losses_B)
    #eG= AA['arr_4']
#%%
Losses[Losses==0] = np.nan
Losses_unk[Losses_unk==0] = np.nan
Losses_kno[Losses_kno==0] = np.nan


#%%
fl.set_plot()


#%%
norm = False
fig = plt.figure()
ax = fig.add_subplot(111)

cos = ['k', 'C0', 'C1']
for il, la in enumerate(labels):
    Lo = Losses_kno[il]
    
    palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

    mm = np.nanmean(np.min(Lo[-50:,:,:], 0), -1)
    ss = np.nanstd(np.min(Lo[-50:,:,:], 0), -1)/np.sqrt(10-np.sum(np.isnan(Lo[-1,:,:]), -1))
    plt.plot(percs, mm, lw=1.5, color=cos[il])
    plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
    
ax.set_ylabel('error recorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_yscale('log')
#ax.set_yticks([0.001, 0.1, 10])
#ax.set_ylim(0.001, 130)
ax.set_xscale('log')
plt.ylim([-2, 100])
plt.yticks([0, 50, 100])
if labels[-1]=='worst_':
    plt.savefig('Figs/Fig5_H.pdf')
else:
    plt.savefig('Figs/Fig5_H_magic.pdf')

#%%
fig = plt.figure()
ax = fig.add_subplot(111)

cos = ['k', 'C0', 'C1']
for il, la in enumerate(labels):
    Lo = Losses_kno[il]
    
    palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

    mm = np.nanmean(np.min(Lo[-50:,:,:], 0), -1)
    ss = np.nanstd(np.min(Lo[-50:,:,:], 0), -1)/np.sqrt(10-np.sum(np.isnan(Lo[-1,:,:]), -1))
    plt.plot(percs, mm, lw=1.5, color=cos[il])
    plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
    
ax.set_ylabel('error recorded act.')
ax.set_xlabel('rec. units M')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
ax.set_yticks([0.001, 0.1, 10, 1000])
#ax.set_ylim(0.001, 130)
ax.set_xscale('log')
#plt.ylim([-2, 100])
#plt.yticks([0, 50, 100])
if labels[-1]=='worst_':
    plt.savefig('Figs/Fig5_Hlog.pdf')
else:
    plt.savefig('Figs/Fig5_Hlog_magic.pdf')

#%%
# #%%
# norm = True
# fig = plt.figure()
# ax = fig.add_subplot(111)

# cos = ['k', 'C0', 'C1']
# for il, la in enumerate(labels):
#     Lo = Losses_kno[il]
    
#     palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

#     mm = np.nanmean(Lo[-1,:,:]/Lo[0,:,:], -1)
#     ss = np.nanstd(Lo[-1,:,:]/Lo[0,:,:], -1)/np.sqrt(10-np.sum(np.isnan(Lo[-1,:,:]), -1))
#     plt.plot(percs, mm, lw=1.5, color=cos[il])
#     plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
    
# ax.set_ylabel('error rec. (norm.)')
# ax.set_xlabel('rec. units M')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_yscale('log')
# ax.set_xscale('log')

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
cos = ['k', 'C0', 'C1']
for il, la in enumerate(labels):
    if il==0 or il==2 or il==1:
        Lo = Losses_kno[il]
        palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
        mm = np.nanmean(Lo[:,:,:], -1)
        plt.plot( mm, lw=1.5, color=cos[il])
    #plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
    
ax.set_ylabel('error rec.')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
if labels[-1]=='worst_':
    plt.savefig('Figs/Fig5_loss_vs_epochs.pdf')
else:
    plt.savefig('Figs/Fig5_loss_vs_epochs_Magic.pdf')


#%%
fig = plt.figure()
ax = fig.add_subplot(111)

cos = ['k', 'C0', 'C1']
for il, la in enumerate(labels):
    if il==0 or il==2 or il==1:
        Lo = Losses_unk[il]
        palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))
        mm = np.nanmean(Lo[:,:,:], -1)
        plt.plot( mm, lw=1.5, color=cos[il])
    #plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
    
ax.set_ylabel('error unrec.')
ax.set_xlabel('epochs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yscale('log')
if labels[-1]=='worst_':
    plt.savefig('Fig5_lossUnrec_vs_epochs.pdf')
else:
    plt.savefig('Figs/Fig5_lossUnrec_vs_epochs_Magic.pdf')



#%%
norm = False

palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

fig = plt.figure()
ax = fig.add_subplot(111)
for il, la in enumerate(labels[0:3]):
    
    if il==0:
        Lo = np.dstack((Losses_unk[0], Losses_unk[3], Losses_unk[4],Losses_unk[5]))#np.dstack((Losses_unk[3],Losses_unk[4],Losses_unk[5]))
    else:
        Lo = Losses_unk[il]
    
    mm = np.nanmean(np.nanmean(Lo[-10:,:,:], 0), -1)
    ss = np.nanstd(np.nanmean(Lo[-10:,:,:], 0), -1)/np.sqrt(np.shape(Lo)[-1]-np.sum(np.isnan(Lo[-1,:,:]), -1))
    plt.plot(percs, mm, lw=1.5, color=cos[il])
    plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
        
        # plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5)
        # plt.scatter(pi, np.nanmean(dat), s=30, edgecolor='k', color=palette[ip], zorder=4)
        # if ip<len(percs)-1:
        #     plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
        
ax.set_xlabel('rec. units M')
ax.set_ylabel('error unrec. act.')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylim([-2, 100])
plt.yticks([0, 50, 120])
#ax.set_yscale('log')
#ax.set_yticks([0.001, 0.1, 100]) g
#ax.set_ylim(0.001, 2000)
ax.set_xscale('log')

if labels[-1]=='worst_':
    plt.savefig('Figs/Fig5_I.pdf')
else:
    plt.savefig('Figs/Fig5_I_magic.pdf')


plt.savefig('Figs/F5lossAll_singleneuron_unrec_final.pdf', transparent=True)

# #%%
# fig = plt.figure()
# ax = fig.add_subplot(111)
# Lo_r = np.dstack((Losses_unk[0], Losses_unk[3], Losses_unk[4],Losses_unk[5]))
# Lo_r = np.nanmean(Lo_r[-10:,:,:], 0)
# Lo_b = Losses_unk[1]
# Lo_b = np.nanmean(Lo_b[-10:,:,:], 0)


# # plt.plot(percs, Lo_r[:,0:10]/Lo_b, -1, c='k', alpha=0.2)
# # plt.plot(percs, Lo_r[:,10:20]/Lo_b, -1, c='k', alpha=0.2)
# # plt.plot(percs, Lo_r[:,20:30]/Lo_b, -1, c='k', alpha=0.2)
# # plt.plot(percs, Lo_r[:,30:40]/Lo_b, -1, c='k', alpha=0.2)

# plt.plot(percs, np.nanmean(Lo_r[:,0:10]/Lo_b, -1), c='k', alpha=0.5)
# plt.plot(percs, np.nanmean(Lo_r[:,10:20]/Lo_b, -1), c='k', alpha=0.5)
# plt.plot(percs, np.nanmean(Lo_r[:,20:30]/Lo_b, -1), c='k', alpha=0.5)
# plt.plot(percs, np.nanmean(Lo_r[:,30:40]/Lo_b, -1), c='k', alpha=0.5)
# plt.plot(percs, percs/percs, c='k', lw=2)

# plt.xscale('log')
# #plt.ylim([0, 50])
# plt.yscale('log')

# ax.set_xlabel('rec. units M')
# ax.set_ylabel('error random/best')

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')

# plt.savefig('Figs/F5lossAll_singleneuron_unrec_paired.pdf', transparent=True)
# #%%
# norm = False

# palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for il, la in enumerate(labels):
#     Lo = Losses_unk[il]

#     mm = np.nanmean(np.min(Lo[-50:,:,:], 0), -1)
#     ss = np.nanstd(np.min(Lo[-50:,:,:], 0), -1)/np.sqrt(10-np.sum(np.isnan(Lo[-1,:,:]), -1))
#     plt.plot(percs, mm, lw=1.5, color=cos[il])
#     plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
        
#         # plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5)
#         # plt.scatter(pi, np.nanmean(dat), s=30, edgecolor='k', color=palette[ip], zorder=4)
#         # if ip<len(percs)-1:
#         #     plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
        
# ax.set_xlabel('rec. units M')
# ax.set_ylabel('error unrec. act.')

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# plt.ylim([-2, 100])
# plt.yticks([0, 50, 150])
# #ax.set_yscale('log')
# #ax.set_yticks([0.001, 0.1, 100]) g
# #ax.set_ylim(0.001, 2000)
# ax.set_xscale('log')

# if labels[-1]=='worst_':
#     plt.savefig('Figs/F5lossAll_unrec_final.pdf', transparent=True)
# else:
#     plt.savefig('Figs/F5lossAll_unrec_final_magic.pdf', transparent=True)



# #%%
# norm = False

# palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for il, la in enumerate(labels):
#     Lo = Losses_unk[il]

#     mm = np.nanmean(np.min(Lo[-50:,:,:], 0), -1)
#     ss = np.nanstd(np.min(Lo[-50:,:,:], 0), -1)/np.sqrt(10-np.sum(np.isnan(Lo[-1,:,:]), -1))
#     plt.plot(percs, mm, lw=1.5, color=cos[il])
#     plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
        
#         # plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5)
#         # plt.scatter(pi, np.nanmean(dat), s=30, edgecolor='k', color=palette[ip], zorder=4)
#         # if ip<len(percs)-1:
#         #     plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
        
# ax.set_xlabel('rec. units M')
# ax.set_ylabel('error unrec. act.')

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# #plt.ylim([-2, 100])
# #plt.yticks([0, 50, 150])
# ax.set_yscale('log')
# #ax.set_yticks([0.001, 0.1, 100]) 
# #ax.set_ylim(0.001, 2000)
# ax.set_xscale('log')

# if labels[-1]=='worst_':
#     plt.savefig('Figs/F5lossAll_unrec_final_log.pdf', transparent=True)
# else:
#     plt.savefig('Figs/F5lossAll_unrec_final_log_magic.pdf', transparent=True)




# #%%

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for il, la in enumerate(labels):
#     Lo = Losses_unk[il]
#     for ip, pi in enumerate(percs):
#         plt.scatter(pi*np.ones(np.shape(Lo)[-1]), Lo[-1,ip,:], color=cos[il])
# ax.set_xscale('log')    
# ax.set_yscale('log')    

# # #%%
# # norm = False

# # palette = plt.cm.plasma(np.linspace(0, 0.8, len(percs)))

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for il, la in enumerate(labels):
# #     Lo = Losses_unk[il]

# #     mm = np.nanmean(Lo[-1,:,:]/Lo[0,:,:], -1)
# #     ss = np.nanstd(Lo[-1,:,:]/Lo[0,:,:], -1)/np.sqrt(10)
# #     plt.plot(percs, mm, lw=1.5, color=cos[il])
# #     plt.fill_between(percs, mm-ss, mm+ss, alpha=0.4, color=cos[il]) 
        
# #         # plt.scatter(pi*np.ones(len(dat)), dat, color=palette[ip], alpha=0.5)
# #         # plt.scatter(pi, np.nanmean(dat), s=30, edgecolor='k', color=palette[ip], zorder=4)
# #         # if ip<len(percs)-1:
# #         #     plt.plot([pi, percs[ip+1]], [np.nanmean(dat), np.nanmean(Lo[-1,ip+1,:])], c='k', lw=2)
        
# # ax.set_xlabel('rec. units M')
# # ax.set_ylabel('error unrec. (norm.)')

# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.yaxis.set_ticks_position('left')
# # ax.xaxis.set_ticks_position('bottom')
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# # ax.set_yticks([0.001, 0.1, 10])
# # plt.savefig('Figs/F5lossworst_singleneuron_unrec_final.pdf', transparent=True)

# #%%
# # norm = False
# # Lo = losses_G
# # palette = plt.cm.plasma(np.linspace(0, 0.8, len(used_percs)))
# # fig = plt.figure(figsize=[3.5, 2.7])
# # ax = fig.add_subplot(211)
# # for ip, pi in enumerate(used_percs):
# #     if norm:
# #         mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
# #         ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
# #     else:
# #         mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
# #         ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
# #     xx = np.arange(len(mm))*10
# #     plt.plot(xx, mm, c=palette[ip])
# #     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
# # ax.set_ylabel('error gain')
# # #ax.set_xlabel('epochs')
# # ax.set_xticklabels([])
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.yaxis.set_ticks_position('left')
# # ax.xaxis.set_ticks_position('bottom')
# # ax.set_ylim([0, 1.2])
# # ax.set_yticks([0, 1.0])
# # ax.set_yticklabels(['0.0', '1.0'])
# # plt.savefig('Figs/F5lossworst_singleneuron_gains.pdf', transparent=True)

# # norm = False
# # Lo = losses_B
# # ax = fig.add_subplot(212)
# # for ip, pi in enumerate(used_percs):
# #     if norm:
# #         mm = np.nanmean((Lo[:,ip,:]/Lo[0,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
# #         ss = np.nanstd((Lo[:,ip,:]/Lo[0,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]
# #     else:
# #         mm = np.nanmean((Lo[:,ip,:]),-1)#/np.nanmean(Lo[0,ip,:], -1)
# #         ss = np.nanstd((Lo[:,ip,:]), -1)/np.sqrt(ics)#/Lo[0,ip,:]        
# #     xx = np.arange(len(mm))*10
# #     plt.plot(xx, mm, c=palette[ip])
# #     plt.fill_between(xx, mm-ss, mm+ss, color=palette[ip], alpha=0.2)
# # ax.set_ylabel('error bias')
# # ax.set_xlabel('epochs')
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.yaxis.set_ticks_position('left')
# # ax.xaxis.set_ticks_position('bottom')
# # #ax.set_ylim([0, 7])
# # ax.set_ylim([0, 0.7])
# # #ax.set_yscale('log')
# # plt.savefig('Figs/F5lossworst_singleneuron_biases.pdf', transparent=True)
# #%%
# dtype = torch.FloatTensor 
# input_train, output_train, cond_train = fl.create_input(taus, trials)
# input_train = torch.from_numpy(input_train).type(dtype)
# output_train = torch.from_numpy(output_train).type(dtype)
# #%%
# # Plot error
# ic  = 0 #6 is good
# IPS = np.arange(len(used_percs))[0:1]#[3, 5, 6, 7, 8, 12, 14]
# magic = False
# best = False
# for ic in range(10):
#     for IP in IPS:
    
#         tri = 2
#         print(used_percs[IP])
#         if best:
#             AA = np.load(stri+'task_best_losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(0.001)+'_unpacked.npz')
#         elif not best:
#             AA = np.load(stri+'task_worst_losses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(used_percs[IP])+'_lr_'+str(0.005)+'_unpacked.npz')
#         e_g = AA['arr_5']
#         e_b = AA['arr_6']
#         e_gf = AA['arr_7']
#         e_bf = AA['arr_8']
        
#         #f_net = np.load('example_net_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
#         J = f_net['arr_0']
#         gt = f_net['arr_1']
#         bt = f_net['arr_2']
#         wI = f_net['arr_3']
#         wout = f_net['arr_4']
#         x0 = f_net['arr_5']
#         mwrec_ = f_net['arr_6']
#         refEI_ = f_net['arr_7']
#         refVec = refEI_[0,:]
    
#         f_seq = np.load('sequences_example_net_softplus_2.npz')

#         Ridcs = f_seq['arr_0']
#         idcs = f_seq['arr_1']
#         idcs_w = f_seq['arr_2']
#         Ridcs_w = f_seq['arr_3']
        
#         if magic and best:
#             seq = Ridcs
#         elif not magic and best:
#             seq = idcs
#         elif not magic and not best:
#             seq = idcs_w
#         elif magic and not best:
#             seq = Ridcs_w
#         seq = seq.astype(np.int16)
        
#         J = J[seq,:]
#         J = J[:,seq]
#         gt = gt[seq]
#         bt = bt[seq]
#         wI = wI[:,seq]
#         wout = wout[seq]
#         x0 = x0[seq]
#         mwrec_ = mwrec_[seq,:]
#         mwrec_ = mwrec_[:,seq]
#         refEI = refEI_[seq,:]
#         refEI = refEI[:,seq]
#         refVec0 = refEI[0,:]
#         refVec = refVec0[seq]
#         input_size = np.shape(wI)[0]
#         N = np.shape(wI)[1]
#         hidden_size = N
        
#         # gtr = e_g[:,-1]+gt
#         # btr = e_b[:,-1]+bt
#         gtr = gt+e_gf#+e_g[:,-1]
#         btr = bt+e_bf#+e_b[:,-1]
#         us = np.linalg.eigvals(J.dot(np.diag(gtr)))
#         print(np.max(us.real))
        
#         g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
#         g_Tra  = torch.from_numpy(gtr[:,np.newaxis]).type(dtype)
#         b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
#         b_Tra  = torch.from_numpy(btr[:,np.newaxis]).type(dtype)
        
#         wi_init = torch.from_numpy(wI).type(dtype)
#         wrec= torch.from_numpy(J).type(dtype)
#         H0 = torch.from_numpy(x0).type(dtype)
#         mwrec = torch.from_numpy(mwrec_).type(dtype)
#         refEI = torch.from_numpy(refEI).type(dtype)
#         wo_init2 = np.eye(N)
#         wo_init2 = torch.from_numpy(wo_init2).type(dtype)
        
#         Net_True = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
#                                   train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
#         NetJ2    = fl.RNN(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_Tra, g_Tra, h0_init=H0,
#                                   train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
#         Out_True = Net_True(input_train)
#         out_true = Out_True.detach().numpy()
        
#         Out_Tra = NetJ2(input_train)
#         out_tra = Out_Tra.detach().numpy()
        
#         # output_trainall = np.copy(out_true)
#         # plt.plot(np.dot(wout, output_trainall[10,:,:].T), c='k')
        
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
#         plt.savefig('Figs/F5lossworst_example_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
#         plt.show()
    
    
#         fig = plt.figure(figsize=[3.5, 3.2])
#         ax = fig.add_subplot(111)
#         plt.imshow(np.abs(out_tra[tri,::selecT,selecN]), cmap='Greys', vmin=0, vmax=50, origin='lower', rasterized=True)
#         rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor='k', alpha=0.5, facecolor='none')
#         ax.add_patch(rect)
#         rect = patches.Rectangle((1, 1),  len(taus[::selecT])-2.5,used_percs[IP]-2, linewidth=1.5, edgecolor=palette[IP], alpha=0.9, facecolor='none')
#         ax.add_patch(rect)
#         ax.set_ylabel('neuron id')
#         ax.set_xlabel('time')
#         ax.set_xticks([0, 100])#labels('')
#         ax.set_xticklabels(['0', '15'])
#         plt.savefig('Figs/F5lossworst_exampleAct_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
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
            
            
#             plt.savefig('Figs/F5lossworst_example_legend.pdf', dpi=300, transparent=True)
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
#         plt.savefig('Figs/F5lossworst_readout_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
#         plt.show()

#     #%%
    

#     fig = plt.figure(figsize=[1.5*2.2, 1.5*2*0.7])
#     ax = fig.add_subplot(111)
#     dat =e_g[0:used_percs[IP],-1]
    
#     sd = 0.05
    
#     val = 0.
#     plt.scatter(sd*np.random.randn(len(dat)), dat, edgecolor='k', linewidth=0.3, color='C5', rasterized=True)    
#     parts = ax.violinplot(
#         dat, showmeans=False, showmedians=False,
#         showextrema=False, positions=[val])
#     for pc in parts['bodies']:
#         pc.set_edgecolor('black')
#         pc.set_facecolor('C5')
#         pc.set_alpha(0.5)
#     plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#     plt.scatter([val], [np.mean(dat)], s=50, color='C5', edgecolor='k', zorder=3)    

#     val = 0.4
#     dat =e_g[used_percs[IP]:,-1]
    
#     parts = ax.violinplot(
#         dat, showmeans=False, showmedians=False,
#         showextrema=False, positions=[val])
#     for pc in parts['bodies']:
#         pc.set_edgecolor('black')
#         pc.set_alpha(0.5)
#         pc.set_facecolor(np.ones(3)*0.3)
#     plt.scatter(val+sd*np.random.randn(len(dat)), dat, edgecolor=0.1*np.ones(3), color='w', linewidth=0.3, rasterized=True)  
#     plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#     plt.scatter([val], [np.mean(dat)], s=50, color='w', edgecolor='k', zorder=3)  
        
    
#     val = 1.1
#     dat =e_b[0:used_percs[IP],-1]

#     parts = ax.violinplot(
#         dat, showmeans=False, showmedians=False,
#         showextrema=False, positions=[val])
#     for pc in parts['bodies']:
#         pc.set_edgecolor('black')
#         pc.set_alpha(0.5)
#         pc.set_facecolor('C6')
#     plt.scatter(val+sd*np.random.randn(len(dat)), dat, edgecolor='k', color='C6', linewidth=0.3, rasterized=True)    
#     plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#     plt.scatter([val], [np.mean(dat)], s=50, color='C6', edgecolor='k', zorder=3)          
    
#     val = 1.5
#     dat =e_b[used_percs[IP]:,-1]
 
#     parts = ax.violinplot(
#         dat, showmeans=False, showmedians=False,
#         showextrema=False, positions=[val])
#     for pc in parts['bodies']:
#         pc.set_edgecolor('black')
#         pc.set_alpha(0.5)
#     plt.scatter(val+sd*np.random.randn(len(dat)), dat, edgecolor='k', color='w', linewidth=0.3, rasterized=True)   
#     plt.plot([val, val], [np.mean(dat)-np.std(dat), np.mean(dat)+np.std(dat)], c='k', lw=4)
#     plt.scatter([val], [np.mean(dat)], s=50, color='w', edgecolor='k', zorder=3) 
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')    
#     ax.set_xticks([0, 0.4, 1.1, 1.5])
#     if used_percs[IP]==30:
#         ax.set_xticklabels([r'$g_{rec}$', r'$g_{unrec}$', r'$b_{rec}$', r'$b_{unrec}$',], fontsize=14, rotation=30)
#     else:
#         ax.set_xticklabels([r'$g_{rec}$', r'$g_{unrec}$', r'$b_{rec}$', r'$b_{unrec}$',], fontsize=14, rotation=30, color='w')
#     ax.set_ylabel('error')
#     ax.set_yticks([-1, 0, 1])
    
#     plt.savefig('Figs/F5lossworst_errorHist_Nrec_'+str(used_percs[IP])+'_legend.pdf', dpi=300, transparent=True)
#     plt.show()
    
    

# #%%
# fig = plt.figure(figsize=[1.5*2.2*1., 1.5*2*1.])
# ax = fig.add_subplot(111)
# ax.imshow(J, vmin = -0.2, vmax=0.2, cmap='bwr', origin='lower')
# ax.set_xlabel('pre-syn neuron')
# ax.set_ylabel('post-syn neuron')
# ax.set_yticks([0, 100, 200, 300])
# ax.set_xticks([0, 100, 200, 300])
# plt.savefig('Figs/F5lossworst_B_conn.pdf', transparent=True)

# #%%
# Vmin = 0
# Vmax = 15
# fig = plt.figure(figsize=[3, 2])
# ax = fig.add_subplot(111)
# bl = plt.pcolor(taus, np.arange(N), ((out_tra[0,:,:]).T), vmin=Vmin, vmax=Vmax, cmap = 'Greys', shading='auto', rasterized=True)
# fig.colorbar(bl, ticks = [ 0, ])
# ax.set_ylabel('neuron index')
# ax.set_xlabel('time')
# ax.set_xticks([0, 15])
# ax.set_xticklabels(['0', '15'])
# ax.set_yticks([0, 120, 300])
# ax.set_yticklabels(['1', 'M', 'N'])
# ax.set_xticklabels(['0', 'T'])
# plt.savefig('Figs/F5lossworst_A_act.pdf', dpi=300, transparent=True)



