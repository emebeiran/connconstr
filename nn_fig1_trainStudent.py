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


#print(sys.argv)
#allseeds = int(sys.argv[2])
#percs = np.array((1, 2, 4, 8,10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 150, 200, 250, 300))#np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40, 80, 150, 200, 250, 300))#np.array((0.0, 1/600, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1,  0.2, 0.5,  1.0))

count = 0
ic_ = int(int(sys.argv[1])-1)
# for ip_ in range(len(percs)):
#     for ic_ in range(allseeds):
#         if int(sys.argv[1])==(count+1):
#             ic = ic_
#             ip=ip_
#             print('seed: '+str(ic))
#             print('output: '+str(percs[ip]))
#             print('count'+str(count))
#         count+=1
        
for ic in [ic_,]:#range(1):
    # ic=ic+1
    f_net = np.load('example_netGain_softplus_2.npz')#np.load('example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
    J = f_net['arr_0']
    gt = f_net['arr_1']
    bt = f_net['arr_2']
    wI = f_net['arr_3']
    wout = f_net['arr_4']
    x0 = f_net['arr_5']
    mwrec_ = f_net['arr_6']
    refEI = f_net['arr_7']
    refVec0 = refEI[0,:]

    alpha = dta
    np.random.seed(ic)
    # Sorting
    # if ic==0:
    #     seq = np.arange(N)#np.random.permutation(np.arange(N))
    # else:
    seq = np.random.permutation(np.arange(N))
    
    seq = seq.astype(np.int16)
        
    
    J = J[seq,:]
    J = J[:,seq]
    
    gt = gt[seq]
    bt = bt[seq]
    
    wI = wI[:,seq]
    wout = wout[seq]
    x0 = x0[seq]
    mwrec_ = mwrec_[seq,:]
    mwrec_ = mwrec_[:,seq]
    refEI = refEI[seq,:]
    refEI = refEI[:,seq]
    refVec = refVec0[seq]
    
    
    
    # ic= 0 #
    # ip = 2 #
    
    g_factor = 0.5 #how much to rescale initial gains w.r.t. true gains (for stability)
    lr = 0.001#lrs[ilr]
    print(lr)
    dtype = torch.FloatTensor 
    
    

    
    input_size = np.shape(wI)[0]
    N = np.shape(wI)[1]
    hidden_size = N
    
    #%%
    g_Ran = np.copy(gt)#g_factor*np.random.permutation(gt)#sig_g*np.random.randn(N,1)+1
    g_Ran[refVec>0] = np.random.permutation(g_Ran[refVec>0])
    g_Ran[refVec<0] = np.random.permutation(g_Ran[refVec<0])
    
    g_Ran = g_factor*g_Ran
    #g_Ran = g_Ran[seq] 
    g_init_Ran = torch.from_numpy(g_Ran[:,np.newaxis]).type(dtype)
    g_True = torch.from_numpy(gt[:,np.newaxis]).type(dtype)
    
    
    b_Ran = np.copy(bt)#np.random.permutation(bt)#sig_g*np.random.randn(N,1)+1
    b_Ran[refVec>0] = np.random.permutation(b_Ran[refVec>0])
    b_Ran[refVec<0] = np.random.permutation(b_Ran[refVec<0])
    
    b_init_Ran = torch.from_numpy(b_Ran[:,np.newaxis]).type(dtype)
    b_True = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
    
    
    wi_init = torch.from_numpy(wI).type(dtype)
    wrec= torch.from_numpy(J).type(dtype)
    H0 = torch.from_numpy(x0).type(dtype)
    
    # These two arrays were used for training the ground truth. Not really used here
    mwrec = torch.from_numpy(mwrec_).type(dtype)
    
    
    refEI = torch.from_numpy(refEI).type(dtype)
    
    n_ep = 7000#7000#500
    
    rep_n_ep = 1#16#32
    
    wo_init2 = np.eye(N)
    Wo_init2 = torch.from_numpy(wo_init2).type(dtype)
    
    wo_iniT = torch.from_numpy(wout[:,np.newaxis]).type(dtype)
    
    input_Train = torch.from_numpy(input_train).type(dtype)
    output_Train = torch.from_numpy(output_train).type(dtype)
    
    
    Nal = 1#int(percs[ip])
    Ntot = Nal
    #%%
    try:
        np.load(stri+'task_Readgain4RslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz')
    except:
        #%%
        # Define RNN
        Net_True = fl.RNNgain(input_size, hidden_size, Ntot, wi_init, wo_iniT, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                                  train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
        NetJ2 = fl.RNNgain(input_size, hidden_size, Ntot, wi_init, wo_iniT, wrec, mwrec, refEI, b_init_Ran, g_init_Ran, h0_init=H0,
                                  train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.002, alpha=alpha, linear=False)
        
        #%%
        output_TrainAll = Net_True(input_Train).detach()
        output_trainall = output_TrainAll.detach().numpy()
        
        output_Train0All = NetJ2(input_Train).detach()
        output_train0all = output_Train0All.detach().numpy()
        
    
        # #%%
        # trial = 3
        # unit  = 0
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
        #%%
        
        Net_True_ = fl.RNNgain(input_size, hidden_size, N, wi_init, Wo_init2, wrec, mwrec, refEI, b_True, g_True, h0_init=H0,
                                  train_b = False, train_g=False, train_conn = False, train_wout = False, train_h0=False, noise_std=0.00, alpha=alpha, linear=False)
        
        NetJ2_ = fl.RNNgain(input_size, hidden_size, N, wi_init, Wo_init2, wrec, mwrec, refEI, NetJ2.b, NetJ2.g, h0_init=H0,
                                  train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, noise_std=0.00, alpha=alpha, linear=False)
        
        output_TrainAll_ = Net_True_(input_Train).detach()
        output_trainall_ = output_TrainAll_.detach().numpy()
        
        output_All_ = NetJ2_(input_Train).detach()
        output_all_ = output_All_.detach().numpy()
        

        Nal0 = 0
        loss_unk_, loss_kno_, loss_unk_R_, loss_kno_R_ = fl.get_losses(input_train, output_trainall_, taus, NetJ2_, gsJ_, bsJ_, Nal0, calculate=False, gain=True, cc=True, readout=True, readout_wout=wout[:,np.newaxis])
        
        
    
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
        
        np.savez(stri+'task_Readgain4RslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz', losses[::10],  losses_unk, losses_kno, losses_G, losses_B, eG, eB, eGf, eBf, seq, loss_unk_R_, loss_kno_R_)
            
        try:
            np.load(stri+'task_Readgain4RslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz')
        except:
            print('error')
            np.savez(stri+'task_Readgain4RslCVlosses_fig1_N_'+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'_unpacked.npz', losses[::10], losses_unk, losses_kno, losses_G, losses_B, eG, eB, eGf, eBf, seq, loss_unk_R_, loss_kno_R_)

#%%
        torch.save(NetJ2.state_dict(), stri+"net_fig0_4_N_"+str(N)+'_ic_'+str(ic)+'_nRec_'+str(Nal)+'_lr_'+str(lr)+'.pt')
# #%%
# Net = NetJ2_

# bs = NetJ2.b.detach().numpy()[:,0]
# gs = NetJ2.g.detach().numpy()[:,0]

# x0 = Net.h0.detach().numpy()
# J = Net.wrec.detach().numpy()
# refEI = Net.refEI.detach().numpy()
# mwrec = Net.mwrec.detach().numpy()
# wi = Net.wi.detach().numpy()

# N = Net.hidden_size
    
# dta = taus[1]-taus[0]
# x = x0
# tri = np.shape(output_train)[0]
# x = np.zeros((tri, N))

# sort = 0
# out_ = np.zeros((tri, N))
# loss_kno_=0
# loss_unk_=0

# x[0:N,:] = x0

# r = fl.softpl(x+bs)
# c_t = 0
# out_s = np.zeros((tri, len(taus), N))
    
# effW_ = fl.relu(J*refEI)
# effW_ = effW_ * refEI
# effW_ = effW_-J*(np.abs(refEI)-1)

# effW = effW_ * mwrec

# for it, ta in enumerate(taus):

#     out_ = gs*r        
#     out_s[:,it,:] = out_
#     x=x+dta*(-x+(r.dot(np.diag(fl.relu(gs)))).dot(effW.T)+input_train[:,it,:].dot(wi))

#     r = fl.softpl(x+bs)
# #%%
# IT = 10
# plt.plot(out_s[IT,:,0:2], c='C0')
# plt.plot(output_all_[IT,:,0:2], c='C1')
# #%%
# plt.plot(out_s[20,:,0:2]-output_all_[20,:,0:2])
#%%
# plt.plot(lossJ_[::10])
# plt.plot(loss_unk_)
# plt.yscale('log')
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

