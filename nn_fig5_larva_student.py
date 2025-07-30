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
stri = 'Data/' 

seed =21
np.random.seed(seed)

#generate trials
input_train, output_train, cond_train = fl.create_input(taus, trials)


#%%
f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
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
print(sys.argv)
allseeds = int(sys.argv[2])
percs = np.array((1, 2, 4, 8,10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 150, 200, 250, 300))#np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40, 80, 150, 200, 250, 300))#np.array((0.0, 1/600, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1,  0.2, 0.5,  1.0))

count = 0
for ip_ in range(len(percs)):
    for ic_ in range(allseeds):
        if int(sys.argv[1])==(count+1):
            ic = ic_
            ip=ip_
            print('seed: '+str(ic))
            print('output: '+str(percs[ip]))
            print('count'+str(count))
        count+=1
#%% 

# ic= 0
# ip = 2

g_factor = 0.5 #how much to rescale initial gains w.r.t. true gains (for stability)
lr = 0.002#lrs[ilr]

dtype = torch.FloatTensor 


np.random.seed(ic)

input_size = np.shape(wI)[0]
N = np.shape(wI)[1]
hidden_size = N

#%%
g_Ran = np.copy(gt)#g_factor*np.random.permutation(gt)#sig_g*np.random.randn(N,1)+1
g_Ran[refVec>0] = np.random.permutation(g_Ran[refVec>0])
g_Ran[refVec<0] = np.random.permutation(g_Ran[refVec<0])

g_Ran = g_factor*g_Ran
g_init_Ran = torch.from_numpy(g_Ran[:,np.newaxis]).type(dtype)
g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)


b_Ran = np.copy(bt)#np.random.permutation(bt)#sig_g*np.random.randn(N,1)+1
b_Ran[refVec>0] = np.random.permutation(b_Ran[refVec>0])
b_Ran[refVec<0] = np.random.permutation(b_Ran[refVec<0])

b_init_Ran = torch.from_numpy(b_Ran[:,np.newaxis]).type(dtype)
b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)


wi_init = torch.from_numpy(wI).type(dtype)
wrec= torch.from_numpy(J).type(dtype)
#sChi = 0.5
#Chi = sChi * np.random.randn(N,N)/np.sqrt(N)

#%%
# for perss in range(10):
Chi2 = np.copy(J)
p_skconn = 0.005
sig_J= p_skconn
for j in range(N):
    for i in range(N):
        if J[i,j]==0:
            if np.random.rand()>p_skconn:
                continue
            else:
                if refEI[0,j]>0:
                    Chi2[i,j] = np.random.permutation(J[J[:,j]>0,j])[0]
                else:
                    Chi2[i,j] = np.random.permutation(J[J[:,j]<0,j])[0]
        else:
            Chi2[i,j] = J[i,j]*(1+np.random.randn()*sig_J)
wrec_stu= torch.from_numpy(Chi2).type(dtype)                        
        
#     # #%%
#     us = np.linalg.eigvals(Chi2)
#     uJ = np.linalg.eigvals(J)
    
#     plt.scatter(np.real(us), np.imag(us), s=4, c='C0')
#     plt.scatter(np.real(uJ), np.imag(uJ), c='k')
# plt.show()

#%%
H0 = torch.from_numpy(x0).type(dtype)

# These two arrays were used for training the ground truth. Not really used here
mwrec = torch.from_numpy(mwrec_).type(dtype)


refEI = torch.from_numpy(refEI).type(dtype)

n_ep = 7000#500

rep_n_ep = 1#16#32

wo_init2 = np.eye(N)
wo_init2 = torch.from_numpy(wo_init2).type(dtype)

input_Train = torch.from_numpy(input_train).type(dtype)
output_Train = torch.from_numpy(output_train).type(dtype)


Nal = int(percs[ip])
Ntot = Nal


#%%
# Define RNN

Net_True = fl.RNNgain(input_size, hidden_size, N, wi_init, wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                          train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False, beta = 1.)

NetJ2 = fl.RNNgain(input_size, hidden_size, Ntot, wi_init, wo_init2[:,0:Ntot], wrec_stu, mwrec, refEI, b_init_Ran, g_init_Ran, h0_init=H0,
                          train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False, beta = 1.0)

#%%
output_TrainAll = Net_True(input_Train).detach()
output_trainall = output_TrainAll.detach().numpy()

# plt.plot(np.dot(wout, output_trainall[10,:,:].T), c='k')
# plt.show()
#%%
# trial = 3
# unit  = 1
# plt.plot(output_trainall[trial,:,unit], c='k')
# output_0 = NetJ2(input_Train).detach()
# output_0 = output_0.detach().numpy()

# plt.plot( output_0[trial,:,unit])

#%%
# Train 

losses_ = []
eG_ = []
eB_ = []

losses = []
eG = []
eB = []

mask_train2 = np.ones((trials, Nt, Ntot))
#mask_train2[:,:,1:Ntot-1] = al*mask_train2[:,:,1:Ntot-1]/N
Mask_train2 = torch.from_numpy(mask_train2).type(dtype) 

for re in range(rep_n_ep):
    print(re)
    lossJ__, bsJ__, gsJ__, outJ_, outJ__ = fl.train(NetJ2, input_Train, output_TrainAll[:,:,0:Nal], Mask_train2, n_epochs=n_ep, plot_learning_curve=False, plot_gradient=False, 
                          lr=lr, clip_gradient = 2., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True)
    if re==0:
        lossJ_ = lossJ__
        gsJ_ = gsJ__
        bsJ_ = bsJ__
    else:
        lossJ_ = np.hstack((lossJ_, lossJ__))
        gsJ_ = np.hstack((gsJ_, gsJ__))
        bsJ_ = np.hstack((bsJ_, bsJ__))
        
e_gsJ_ = np.zeros_like(gsJ_)
e_bsJ_ = np.zeros_like(gsJ_)


for ep in range(np.shape(gsJ_)[1]):
    e_gsJ_[:,ep] = fl.relu(gsJ_[:,ep])-fl.relu(gt)
    e_bsJ_[:,ep] = bsJ_[:,ep]-bt
    
loss_unk_, loss_kno_ = fl.get_losses(input_train, output_trainall, taus, NetJ2, gsJ_, bsJ_, Nal, calculate=False, gain=True)


#%%
losses = lossJ_
#losses_read = loss_re_
losses_kno = loss_kno_
losses_unk = loss_unk_

losses_G = np.sqrt(np.mean(e_gsJ_**2,0))[::10]
losses_B = np.sqrt(np.mean(e_bsJ_**2,0))[::10]
eG = e_gsJ_[:,::100]
eB = e_bsJ_[:,::100]
eGf = e_gsJ_[:,-1]
eBf = e_bsJ_[:,-1]

np.savez(stri+'task_lossesMiCo_Dec'+str(p_skconn)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz', losses[::10],  losses_unk, losses_kno, losses_G, losses_B, eG, eB, eGf, eBf, Chi2)
    
# try:
#     np.load(stri+'task_lossesMiCo_Dec'+str(p_skconn)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz')
# except:
#     print('error')
#     np.savez(stri+'task_lossesMiCo_Dec'+str(p_skconn)+'_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz', losses[::10], losses_unk, losses_kno, losses_G, losses_B, eG, eB, eGf, eBf, J+Chi)

#%%
#torch.save(NetJ2.state_dict(), stri+"net_fig1_N_"+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'.pt')

#%%

#%%
# gsJ_[:,0] = gt
# bsJ_[:,0] = bt


# #%%
# tri = 10
# IX = 130
# plt.figure()
# plt.plot(output_trainall[tri,:,IX], c='k')
# plt.plot(out_s[tri,:,IX])
# plt.show()



#%%

