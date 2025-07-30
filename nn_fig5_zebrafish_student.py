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
    groups = [0,92,134]#[0, 46, 50, 70, 92, 134]
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
AA = np.load('zebrafish.npz')
W_ = AA['arr_0']
I_ = AA['arr_1']
v_in_ = AA['arr_2']
dt_ = AA['arr_3']


N_ = np.shape(W_)[0]
lent_ = len(I_)

r_ = np.zeros((N_,lent_))
for it in range(lent_-1):
    r_[:,it+1] = r_[:,it]+dt_*(-r_[:,it]+W_.dot(r_[:,it])+I_[it]*v_in_)

#%%
plt.imshow(W_, vmin=-0.01, vmax=0.01, cmap='bwr')
plt.show()

plt.plot(np.sum(np.abs(W_),0))
plt.plot(np.sum(np.abs(W_),1))
plt.show()

#%%
maskn = np.sum(np.abs(W_),0)>0#(np.sum(np.abs(W_),0)+np.sum(np.abs(W_),1))>0
N = np.sum(maskn)
def old2new(I):
    i_old = np.arange(N_)
    i_olds = i_old[maskn]
    aa = np.where(i_olds==I)
    return(aa[0][0])
#272, 273, 281
#%%
W = W_[maskn,:]
W = W[:, maskn]
v_in = v_in_[maskn]


#%%
dt = 0.1
T = dt_*len(I_)

time = np.arange(0,T, dt)
time_ = np.arange(0,T, dt_)

I = np.zeros_like(time)
t0s = [1, 8, 15]
for t0 in t0s:
    ai = np.argmin(np.abs(time-t0))
    I[ai] = 10.5/dt
#%%
N = np.shape(W)[0]
lent = len(I)

r = np.zeros((N,lent))
for it in range(lent-1):
    r[:,it+1] = r[:,it]+dt*(-r[:,it]+W.dot(r[:,it])+I[it]*v_in)

#%%
neurs = [272, 273, 281]
for iN in neurs:

    plt.plot(time_, r_[iN,:])
    iN2 = old2new(iN) 
    plt.plot(time, r[iN2,:], '--', c='k')
plt.show()
g = np.random.lognormal(0, 0.3, size=N)

W = W.dot(np.diag(1./g))

#np.savez('zebrafish2.npz', W, I, g, v_in, dt, maskn, N_)

#%%

AA = np.load('zebrafish2.npz')

JJ = AA['arr_0']
I = AA['arr_1']
gf = AA['arr_2']
#bf = AA['arr_2']
x0 = np.zeros_like(gf)#AA['arr_3']
wI = AA['arr_3']
maskn = AA['arr_5']
alpha = AA['arr_4']
N_ = np.exp(AA['arr_6'])

input_size = 1
hidden_size = np.shape(JJ)[0]
output_size = np.shape(JJ)[0]
wI2 = wI[np.newaxis,:]
Beta = 5.
N = hidden_size
wout = np.eye(hidden_size)[:,0:output_size]

#%%
# sig_b = np.std(bf)
sig_g = np.std(gf)

    
#%%
seed = 21
sd = 0.001
n_ep = 210
np.random.seed(seed)


dtype = torch.FloatTensor  
wi_init = torch.from_numpy(wI).type(dtype)
wo_init = torch.from_numpy(wout[:,np.newaxis]).type(dtype)

wrec_init = torch.from_numpy(JJ).type(dtype)
b_init_True= torch.from_numpy(0*gf[:,np.newaxis]).type(dtype)

b_init_Ran= torch.from_numpy(0*gf[:,np.newaxis]).type(dtype)
b_init    = torch.from_numpy(0*gf[:,np.newaxis]).type(dtype)



H0= torch.from_numpy(x0).type(dtype)
#%%
trials = 1
Trials= trials
dta = alpha

TT = len(I)*dta
Nt = len(I)

taus = np.arange(0, TT, dta)
output_train = np.zeros((trials, Nt, output_size))
output_train2 = np.zeros((trials, Nt, hidden_size))
inps = np.zeros((trials, Nt, input_size))
inps[0,:,0]=I

for tr in range(trials):
    #print(tr)
    
    xB = np.zeros((len(taus), hidden_size))
    
    xB[0,:] = x0
    output_train[tr,0,:] = wout.T.dot(x0)
    
    output_train2[tr,0,:]= x0
    
    for it, ta in enumerate(taus[:-1]):
                                 
        xB[it+1,:]=xB[it,:]+(dta)*(-xB[it,:]+JJ.dot(gf*xB[it,:]) + inps[tr,it,:].dot(wI2))
        output_train[tr,it+1,:] = wout.T.dot(xB[it+1,:])
        output_train2[tr,it+1,:] = wout.T.dot(xB[it+1,:])
        
#%%
TR = 0
plt.imshow(output_train[TR,:,:].T, vmin = 0, vmax=4.)
plt.show()

# # plt.imshow(output_train2[TR,:,0:49].T, vmin = 0, vmax=4.)
# # plt.show()
# # plt.plot(inps[TR,:,-2:])
# # plt.show()

#%%
ts = taus

N = np.shape(JJ)[0]
gt = np.random.permutation(gf)


#output_size = N
#wout = np.zeros((N,output_size))

winp = np.zeros((input_size,N))
winp[0,:]= wI2
dtype = torch.FloatTensor 
#Wr_ini =JJ
#Wr_ini_log = np.copy(JJ)
#Wr_ini_log[np.abs(Wr_ini_log)>0] = np.log(np.abs(Wr_ini_log[np.abs(Wr_ini_log)>0] ))

wrec_init = torch.from_numpy(JJ).type(dtype)

g_init = torch.from_numpy(gt[:,np.newaxis]).type(dtype)


wi_init = torch.from_numpy(winp).type(dtype)
wo_init = torch.from_numpy(wout).type(dtype)

dta = ts[1]-ts[0]
Tau = 1.

alpha = dta/Tau
hidden_size=N


h0_init = torch.from_numpy(x0).type(dtype)
input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(output_train)
m = np.ones_like(output_train)
Beta = 5.
lin_var = False

Net = fs.RNN_fish(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init,  g_init, 
          h0_init=h0_init, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.0,
          alpha=alpha,  train_h0=False)
    # class RNN_fish(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, 
    #                   g_init, h0_init = None,  train_g=True, train_conn = False, 
    #                  train_wout = False, train_wi = False, train_h0=False, train_taus = True, noise_std=0.005, alpha=0.2,
    #                   linear=True, beta=1.):
            
#inps[:,:,0:46] = 0.
input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(output_train)
mask_train = torch.from_numpy(m).type(dtype)
trainstr=''
out2 = Net.forward(input_Train)
out = out2.detach().numpy()
#%%
plt.plot(out[0,:,0])
plt.plot(output_train[0,:,0])


#%%

# plt.scatter(np.exp(Net.g.detach().numpy()[:,0]), gf)
# plt.show()
# plt.scatter(Net.h0.detach().numpy(), x0)
# plt.show()

# #%%
# Trs = [3, 18]
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
# IT = 17
# #plt.plot(out[IT,:,3+NN]-output_train2[IT,:,NN], c='k')
# plt.plot(out[IT,:,3+NN], c='k')
# plt.plot(output_train2[IT,:,NN], c='C0')
# plt.show()

# #%%
# plt.scatter(np.ravel(out[:,0,3:49]), np.ravel(output_train2[:,0,0:46]))
# plt.plot([0, 4], [0,4])

# #%%
# plt.hist(np.ravel(out[:,0,3:49])- np.ravel(output_train2[:,0,0:46]))


#%%
percs = np.array((1, 2, 4, 8, 12, 16, 24, 32, 48, 56, 64, 96,  128, 160, 196, 228, 260,))

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
    np.load(stri+'taskFishA_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
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
    g_Ran = np.log(random_group(np.exp(Net.g.detach().numpy()), scale=0.5))
    #b_Ran = random_group(Net.b.detach().numpy())
    
    g_init_Ran = torch.from_numpy(g_Ran).type(dtype)
    #b_init_Ran = torch.from_numpy(b_Ran).type(dtype)
    
    n_ep = 2000#500#400
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
    



    
    NetJ2 = fs.RNN_fish(input_size, hidden_size, al, Net.wi,  wo_init2[:,0:al], Net.wrec,  g_init_Ran, h0_init=Net.h0,
                              train_g=True, train_conn = False, train_wout = False, train_h0=False, 
                               noise_std=0.0001, alpha=alpha)
    
    output_TrainAll = torch.from_numpy(output_train2[:,:,rn_ar[0:al]]).type(dtype)     

    mask_train2 = np.ones((trials, Nt, Ntot))
    #mask_train2[:,:,1:Ntot-1] = al*mask_train2[:,:,1:Ntot-1]/N
    Mask_train2 = torch.from_numpy(mask_train2).type(dtype) 
    
    #lr = 0.01#0.05
    Out = NetJ2(input_Train)
    out_try0 = Out.detach().numpy()
    
    torch.save(NetJ2.state_dict(), stri+"netFish_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'00.pt')

    #%%
    for re in range(rep_n_ep):
        print(re)
        lossJ__,  gsJ__,  outJ_, outJ__ = fs.train_fish(NetJ2, input_Train, output_TrainAll, Mask_train2, n_epochs=n_ep, plot_learning_curve=False, plot_gradient=False, 
                              lr=lr, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True, batch_size=1)
        if re==0:
            lossJ_ = lossJ__
            gsJ_ = gsJ__


        else:
            lossJ_ = np.hstack((lossJ_, lossJ__))
            gsJ_ = np.hstack((gsJ_, gsJ__))

    e_gsJ_ = np.zeros_like(gsJ_)

    
    for ep in range(np.shape(gsJ_)[1]):
        e_gsJ_[:,ep] = np.exp(gsJ_[:,ep])-np.exp(gf)

        
    #%%
    Out = NetJ2(input_Train)
    out_try = Out.detach().numpy()
    
    # plt.plot(out_try[0,:,0])
    # plt.plot(output_TrainAll[0,:,0])    
    # plt.show()
    # #%%
    # plt.plot(lossJ__)
    # plt.show()
    # #%%
    # Out = NetJ2(input_Train)
    # out_try = Out.detach().numpy()
    # #%%
    # TR = 0
    # plt.plot(out_try[TR,:,0:2])
    # plt.plot(out_try0[TR,:,0:2], '--')
    # plt.plot(output_TrainAll[TR,:,0:2], '--', c='k')    
    # #%%
    # TR = 0
    # plt.plot(out_try[TR,:,-12:-10])
    # plt.plot(out_try0[TR,:,-12:-10], '--')
    # plt.plot(output_TrainAll[TR,:,-12:-10], c='k')    
    
    
    #%%
    TR = 0
    fig=plt.figure(figsize=[10,3])
    ax = fig.add_subplot(131)
    plt.imshow(out_try[TR,:,:].T,aspect='auto', vmin=0, vmax=6)
    
    ax = fig.add_subplot(132)
    plt.imshow(out_try0[TR,:,:].T,aspect='auto', vmin=0, vmax=6)
    
    ax = fig.add_subplot(133)
    plt.imshow(output_TrainAll[TR,:,:].T,aspect='auto', vmin=0, vmax=6)
    plt.show()

    #%%
    Nkno = al
    epos = np.shape(gsJ_)[1]
    loss_kno = np.zeros(epos)
    loss_unk = np.zeros(epos)
    loss_all = np.zeros(epos)
    loss_kno_r2 = np.zeros(epos)
    loss_unk_r2 = np.zeros(epos)
    
    NT2 = hidden_size
    for ep in range(epos):
        if np.mod(ep, 500)==0 or ep ==epos-1:
            print(ep)
    
        output_train_G = np.zeros((trials, Nt, N))
    
        for tr in range(trials):
            #input_train[tr,:,:] = inps[:,tr,:]
            
            xB = np.zeros((len(taus), N))
           
            
            xB[0,:] = x0
            output_train_G[tr,0,:]= xB[0,:]
    
            for it, ta in enumerate(taus[:-1]):
                xB[it+1,:]=xB[it,:]+(alpha)*(-xB[it,:]+JJ.dot(gsJ_[:,ep]*(xB[it,:])) + inps[tr,it,:].dot(wI2))
                output_train_G[tr,it+1,:] = (xB[it+1,:])
    
        loss = (output_train_G[:,:,0:]-output_train2[:,:,0:])**2
        loss = np.mean(np.mean(loss,0),0)
        loss_all[ep] = np.mean(loss)
        
        otG = output_train_G[0,:,0:]#, output_train_G[1,:,0:]))
        ot2 = output_train2[0,:,0:]#, output_train2[1,:,0:]))
        CC = np.corrcoef(otG.T, ot2.T)
        lossr2 = np.diag(CC[0:N,N:])                       

        loss_kno[ep]  = np.mean(loss[rn_ar[0:Nkno]])
        loss_kno_r2[ep]  = 1-np.mean(lossr2[rn_ar[0:Nkno]])
        if Nkno==NT2:
            loss_unk[ep] = np.nan
            loss_unk_r2[ep] = np.nan
        else:
            loss_unk[ep] = np.mean(loss[rn_ar[Nkno:N]])
            loss_unk_r2[ep] = 1-np.mean(lossr2[Nkno:])
            
        if ep==0:
            output_train_G0 = np.copy(output_train_G)
        

#%%    
    losses = lossJ_
    # #%%
    # plt.plot(losses)
    # plt.plot(loss_kno)
    # plt.show()
    
    # #%%
    # Out = NetJ2(input_Train)
    # out_try = Out.detach().numpy()
    
    # NN = 7
    # TT = 0
    # plt.plot(output_train_G0[TT,:,NN], '--', c='C0', label='initial')
    # plt.plot(output_train_G[TT,:,NN], label='final')
    # plt.plot(output_train2[TT,:,NN], c='k', label='target')    

    # plt.legend()
    # plt.show()
    
    # #%%
    # Out = NetJ2(input_Train)
    # out_try = Out.detach().numpy()
    
    # NN = 37
    # TT = 0
    # plt.plot(output_train_G0[TT,-1,:], '--', c='C0', label='initial')
    # plt.plot(output_train_G[TT,-1,:], label='final')
    # plt.plot(output_train2[TT,-1,:], '--', c='k', label='target')    
    # plt.legend()
    # plt.show()
    
    np.savez(stri+'taskFish_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz', losses[::10],  loss_kno, loss_unk, loss_kno_r2, loss_unk_r2)
        
    try:
        np.load(stri+'taskFish_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz')
    except:
        print('error')
        np.savez(stri+'taskFish_lossesAshokAdam_percEI_N_'+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'_unpacked.npz', losses[::10], loss_kno, loss_unk, loss_kno_r2, loss_unk_r2)
    
    #%%
    torch.save(NetJ2.state_dict(), stri+"netFish_SGD_EI_AshokAdam_N_"+str(N)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'_sort_'+str(inf_sort)+'.pt')


