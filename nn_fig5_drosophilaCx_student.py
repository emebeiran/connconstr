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
import functions_trainserverAshokOct24 as fs

def relu(x, beta=5.):
    yy = np.exp(beta*x)
    sol= np.copy(x)    
    sol[yy<100] = (1/beta)*np.log(1+yy[yy<100])
    return sol

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
AA = np.load('params_netSimpleRing2_final.npz')

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
y,m,inps, ts = fs.generate_targetssimple(Trials)
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
        
# #%%
# TR = 1
# plt.imshow(y[TR,:,:].T, vmin = 0, vmax=4.)
# plt.show()

# plt.imshow(output_train2[TR,:,0:49].T, vmin = 0, vmax=4.)
# plt.show()
# plt.plot(inps[TR,:,-2:])
# plt.show()

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
Net.load_state_dict(torch.load(trainstr+'netPopVec_Wrec_simplering2.pt', map_location='cpu'))
out2 = Net.forward(input_Train)
out = out2.detach().numpy()
# #%%
# plt.scatter(np.exp(Net.g.detach().numpy()[:,0]), gf)
# plt.show()
# plt.scatter(Net.h0.detach().numpy(), x0)
# plt.show()

# #%%
# Trs = [10, 12]
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
# IT = 7
# plt.plot(out[IT,:,3+NN]-output_train2[IT,:,NN], c='k')
# plt.plot(out[IT,:,3+NN], c='k')
# plt.plot(output_train2[IT,:,NN], c='C0')
# plt.show()

# #%%
# plt.scatter(np.ravel(out[:,0,3:49]), np.ravel(output_train2[:,0,0:46]))
# plt.plot([0, 4], [0,4])

# #%%
# plt.hist(np.ravel(out[:,0,3:49])- np.ravel(output_train2[:,0,0:46]))

#%%
#%%
percs = np.array((1, 2, 4,  6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, ))

#COUNT = 4
count = 0
allseeds = 20

for seed in range(allseeds):
    for ip_ in range(len(percs)):
        if int(sys.argv[1])==(count+1):
            ilr = 0
            ic = seed
            ip = ip_
            # if iso==0:
            #     inf_sort = False
            # else:
            #     inf_sort = True
                
            print('seed: '+str(ic))
            print('output: '+str(percs[ip]))
            print('count'+str(count))
            
        count+=1

inf_sort = True
#%%
stri = 'Data/'


lrs = [0.001,]#[ 0.005, ] #new
lr = lrs[ilr]
ial=ip
al = percs[ial]
Ntot = int(al)

try:
    np.load(stri+'taskSimpleRingFMix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
except:
    print('not saved')
    # inf_sort = False
    # ic = 0
    # ip = 12
    # ilr= 0
    
    
    #%%
    wo_init2_ = np.eye(hidden_size)
    losses_ = []
    eG_ = []
    
    losses = []
    eG = []
    np.random.seed(ic+10)
    g_Ran = np.log(random_group(np.exp(Net.g.detach().numpy()), scale=0.8))
    b_Ran = random_group(Net.b.detach().numpy())
    
    g_init_Ran = torch.from_numpy(g_Ran).type(dtype)
    b_init_Ran = torch.from_numpy(b_Ran).type(dtype)
    
    n_ep = 5000#500#400
    rep_n_ep = 1#5#5
    
    
    #np.array((0.0, 1/600, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5,  0.75, 0.9, 1.0))
    Ntot = int(al)
    rn_ar = np.random.permutation(np.arange(hidden_size))#np.random.permutation(np.arange(Np))
    print(rn_ar[0:Ntot])
    #rn_ar = np.0arange(Np+Nm)
    # rn_ar[0:5] = np.arange(5,10)
    # rn_ar[5:10] = np.arange(0,5)
    wo_init2 = torch.from_numpy(wo_init2_[:,rn_ar]).type(dtype)
    input_Train = torch.from_numpy(inps).type(dtype)
    



    
    NetJ2 = fs.RNN_fly(input_size, hidden_size, al, Net.wi, Net.si, wo_init2[:,0:al], Net.wrec, Net.mwrec, b_init_Ran, g_init_Ran, Net.taus, h0_init=Net.h0,
                              train_b = True, train_g=True, train_conn = False, train_wout = False, train_h0=False, train_si=False, 
                              train_taus=False, noise_std=0.0001, alpha=alpha, linear=lin_var,  beta=Beta)
    
    output_TrainAll = torch.from_numpy(output_train2[:,:,rn_ar[0:al]]).type(dtype)     

    mask_train2 = np.ones((trials, Nt, Ntot))
    #mask_train2[:,:,1:Ntot-1] = al*mask_train2[:,:,1:Ntot-1]/N
    Mask_train2 = torch.from_numpy(mask_train2).type(dtype) 
    
    #lr = 0.01#0.05
    Out = NetJ2(input_Train)
    out_try = Out.detach().numpy()
    #%%
    TR = 10
    plt.plot(out_try[TR,:,:])
    plt.plot(output_TrainAll[TR,:,:], c='k')    
    
    #%%

    for re in range(rep_n_ep):
        print(re)
        lossJ__, bsJ__, gsJ__, wsJ_, outJ_, outJ__ = fs.train_fly(NetJ2, input_Train, output_TrainAll, Mask_train2, n_epochs=n_ep, plot_learning_curve=False, plot_gradient=False, 
                              lr=lr, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True, batch_size=45)
        if re==0:
            lossJ_ = lossJ__
            gsJ_ = gsJ__
            bsJ_ = bsJ__
        else:
            lossJ_ = np.hstack((lossJ_, lossJ__))
            gsJ_ = np.hstack((gsJ_, gsJ__))
            bsJ_ = np.hstack((bsJ_, bsJ__))
    e_gsJ_ = np.zeros_like(gsJ_)
    e_bsJ_ = np.zeros_like(bsJ_)
    
    for ep in range(np.shape(gsJ_)[1]):
        e_gsJ_[:,ep] = np.exp(gsJ_[:,ep])-np.exp(gf)
        e_bsJ_[:,ep] = bsJ_[:,ep]-bf
        
    #%%
    Out = NetJ2(input_Train)
    out_try = Out.detach().numpy()
    
    plt.plot(out_try[0,:,0])
    plt.plot(output_TrainAll[0,:,0])    
    plt.show()
    #%%
    Out = NetJ2(input_Train)
    out_try = Out.detach().numpy()
    # #%%
    # TR = 31
    # plt.plot(out_try[TR,:,0:2])
    # plt.plot(output_TrainAll[TR,:,0:2], c='k')    


    #%%
    Nkno = al
    factor = 5
    epos = np.shape(gsJ_)[1]
    loss_kno = np.zeros(epos//factor)
    loss_unk = np.zeros(epos//factor)
    loss_all = np.zeros(epos//factor)
    loss_kno_r2 = np.zeros(epos//factor)
    loss_unk_r2 = np.zeros(epos//factor)
    
    NT2 = hidden_size
    epcount = 0
    for ep in range(epos):
        if np.mod(ep, 500)==0:
            print(ep)
        if np.mod(ep, factor)==0:
            output_train_G = np.zeros((trials, Nt, N))
        
            for tr in range(trials):
                #input_train[tr,:,:] = inps[:,tr,:]
                
                xB = np.zeros((len(taus), N))
                rB = np.zeros((len(taus)))
                
                xB[0,:] = x0
                output_train_G[tr,0,:]= np.exp(gsJ_[:,ep])*relu(xB[0,:]+bsJ_[:,ep])
        
                for it, ta in enumerate(taus[:-1]):
                    xB[it+1,:]=xB[it,:]+(alpha)*(-xB[it,:]+JJ.dot(np.exp(gsJ_[:,ep])*relu(xB[it,:]+bsJ_[:,ep])) + inps[tr,it,:].dot(wI2))
                    output_train_G[tr,it+1,:] = np.exp(gsJ_[:,ep])*relu(xB[it+1,:]+bsJ_[:,ep])
        
            loss = (output_train_G[:,:,0:]-output_train2[:,:,0:])**2
            loss = np.mean(np.mean(loss,0),0)
            loss_all[epcount] = np.mean(loss)
            
            # otG = np.vstack((output_train_G[0,:,0:], output_train_G[1,:,0:]))
            # ot2 = np.vstack((output_train2[0,:,0:], output_train2[1,:,0:]))
            # CC = np.corrcoef(otG.T, ot2.T)
            
            CC = 0
            TRI = np.shape(output_train2)[0]
            for ii in range(TRI):
                otG = output_train_G[ii,1:,:]#np.vstack((output_stud[0,1:,0:], output_stud[1,1:,0:]))
                ot2 = output_train2[ii,1:,:]#np.vstack((output_train2[0,1:,0:], output_train2[1,1:,0:]))
                CC += np.corrcoef(otG.T, ot2.T)/TRI
            
            lossr2 = np.diag(CC[0:N,N:])                       
    
            loss_kno[epcount]  = np.mean(loss[rn_ar[0:Nkno]])
            loss_kno_r2[epcount]  = 1-np.mean(lossr2[rn_ar[0:Nkno]])
            if Nkno==NT2:
                loss_unk[epcount] = np.nan
                loss_unk_r2[epcount] = np.nan
            else:
                loss_unk[epcount] = np.mean(loss[rn_ar[Nkno:N]])
                loss_unk_r2[epcount] = 1-np.mean(lossr2[Nkno:])
            epcount +=1

#%%    
    losses = lossJ_
    np.savez(stri+'taskSimpleRingFMix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz', losses[::10],  loss_kno, loss_unk, loss_kno_r2, loss_unk_r2)
        
    try:
        np.load(stri+'taskSimpleRingFMix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
    except:
        print('error')
        np.savez(stri+'taskSimpleRingFMix_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz', losses[::10], loss_kno, loss_unk, loss_kno_r2, loss_unk_r2)
    
    #%%
    torch.save(NetJ2.state_dict(), stri+"netSimpleRingFMix_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'.pt')


