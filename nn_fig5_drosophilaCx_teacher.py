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


import pandas as pd
import time
import random
from math import sqrt

#%%
def softplus(x, beta=1.):
    return (1/beta)*np.log(1+np.exp(beta*x))

def real_relu(x):
    return (x>0)*x

def get_angles(out):
    out = out[:,:,0:2]
    out = out/np.sqrt(np.sum(out**2,-1,keepdims=True))
    angs = np.arctan2(out[:,:,1],out[:,:,0])
    return(angs)

def give_velInp(ts, tau = 1.0, p0 = 0.4, runt0 = 0.5, steps = 2, ptot0 = 0.1):
    inp = np.zeros_like(ts)
    if np.random.rand()>ptot0:
        runt = runt0
        while runt<ts[-1]:
            runtp = runt+np.random.exponential(scale=tau) #set interval
            if np.random.rand()<p0:
                inp[(ts>=runt)*(ts<runtp)] = 0.
            else:
                inp[(ts>=runt)*(ts<runtp)] = 0.5*np.sign(np.random.randn())*(np.random.randint(steps)+1)
            runt = runtp
    return(inp)

def generate_targets(trials, epg_ix, T = 6, dt =0.1, startMask = 3, 
                            s_amp = 0.2, mu_amp=1., factor=1., AvgRate = 1., 
                            wbump = 0.1, stay=False):
    
    x = np.linspace(-1,1, 1000) #create bump
    bump = np.exp(-(x/(3/16))**2) #shape

    ts = np.arange(0,T,dt)
    iMask = np.argmin(np.abs(ts-startMask))
    y = np.zeros((trials,len(ts), 3+46)) # The first three dimensions are sine, cosyne, rate
    m = np.zeros((trials,len(ts), 3+46)) 
    x_new = np.linspace(0,1,len(np.unique(epg_ix)))
    x_old = np.linspace(0,1,len(bump))
    inps_ = np.zeros(( trials, len(ts), 46+2)) # the last two are left and right

        
    for tr in range(trials):

        T1 = T
        ori = np.random.rand()*2*np.pi-np.pi #btwn pi and -pi I think
        i_ori = int((len(x)/2)*ori/np.pi)
        bump_shift = np.roll(bump, i_ori)
        subbump = np.interp(x_new, x_old, bump_shift )
        subbump = AvgRate*subbump/np.mean(subbump)
        
        extInp =give_velInp(ts)
        extInp=extInp
        subbump46 = W_16to46.dot(subbump)
        y[tr,:,3:] = subbump46
        y[tr,0:,0] = W_46to3[0,:].dot(subbump46)
        y[tr,0:,1] = W_46to3[1,:].dot(subbump46)
        y[tr,:,2] = W_46to3[2,:].dot(subbump46)

        
        m[tr,iMask:,:] = 1.
        m[tr,iMask:,2] = 1.
        m[tr,iMask:,3:] = wbump*(1/46.)*m[tr,iMask:,3:]
        inps_[tr,0:5,0:46] = subbump46
        #scale input with a random gain
        inps_[tr,0:5,0:46]=np.max((0.2,(mu_amp+s_amp*np.random.randn())))*inps_[tr,0:5,0:46]
    return y, m, inps_, ts

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, si_init, wo_init, wrec_init, mwrec_init,
                 b_init, g_init, tau_init, h0_init = None, train_b = True, train_g=True, train_conn = False, 
                 train_wout = False, train_wi = False, train_h0=True, train_si=True, train_taus = False, noise_std=0.005, alpha=0.2,
                  linear=False, beta=1.):
        
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_b = train_b
        self.train_g = train_g
        self.train_conn = train_conn
        if linear==True:    
            self.non_linearity = torch.clone#torch.tanh
        else:
            self.non_linearity = torch.nn.Softplus(beta=beta)
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if train_wi:
            self.wi.requires_grad= True
        else:
            self.wi.requires_grad= False
            
        self.si = nn.Parameter(torch.Tensor(input_size,1))
        if train_si:
            self.si.requires_grad= True
        else:
            self.si.requires_grad= False
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
            
        self.mwrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.mwrec.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False
        else:
            self.h0.requires_grad = True
            
        self.taus = nn.Parameter(torch.Tensor(hidden_size,1))
        if not train_taus:
            self.taus.requires_grad = False
        else:
            self.taus.requires_grad = True
        
        self.wout = nn.Parameter(torch.Tensor( hidden_size, output_size))
        if train_wout:
            self.wout.requires_grad= True
        else:
            self.wout.requires_grad= False

        self.b = nn.Parameter(torch.Tensor(hidden_size,1))
        if not train_b:
            self.b.requires_grad = False
            
        self.g = nn.Parameter(torch.Tensor(hidden_size,1))
        if not train_g:
            self.g.requires_grad = False
        
        # Initialize parameters
        with torch.no_grad():
            self.wi.copy_(wi_init)
            self.si.copy_(si_init)
            self.wrec.copy_(wrec_init)
            self.wout.copy_(wo_init)
            self.b.copy_(b_init)
            self.g.copy_(g_init)
            self.taus.copy_(tau_init)
            self.mwrec.copy_(mwrec_init)
            
            
            if h0_init is None:
                self.h0.zero_
                self.h0.fill_(1)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0+self.b.T)
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = torch.mul(torch.exp(self.g.T), r).matmul(self.wout)
        # simulation loop
        Max = 5.
        Min = 0.2
        InvTau = (1./(0.5*(Max+Min)+0.5*(Max-Min)*torch.tanh(self.taus)))
        Jmat = torch.exp(self.wrec).mul(self.mwrec)
        #print(InvTau)
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha *(-h + torch.mul(torch.exp(self.g.T), r).matmul(Jmat.t())+ input[:,i,:].matmul(torch.exp(self.si).mul(self.wi))).matmul(InvTau[:,0].diag())
            r = self.non_linearity(h+self.b.T)
            output[:,i+1,:] = torch.mul(torch.exp(self.g.T), r).matmul(self.wout)
        
        return output

def give_taus(taus):
    Max = 5.
    Min = 0.2
    return((0.5*(Max+Min)+0.5*(Max-Min)*np.tanh(taus)))

def set_axis(fig=0, num = 0):
    if fig==0:
        ax = plt.figure().add_subplot(111)
    else:
        ax = fig.add_subplot(num)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom') 
    return(ax)    

def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    
    # Compute loss for each (trial, timestep) (average accross output dimensions)    
    
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, :].mean(dim=-1).sum(dim=-1)#mask[:, :, 0].sum(dim=-1)
    
    return loss_by_trial.mean()

def net_loss(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, cuda=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)
    print(mask.shape())
    print(target.shape())
    print(net(input).shape())
    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
    return(initial_loss.item())

def train(net, _input, _target, _mask, n_epochs, hidden_size, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, keep_best=False, cuda=False, save_loss=False, save_params=True, verbose=True, adam=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    print("Training...")
    if adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)#
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    bs = np.zeros((hidden_size, n_epochs))
    gs = np.zeros((hidden_size, n_epochs))
    # graD = np.zeros((hidden_size, n_epochs))
    
    wr = np.zeros((hidden_size, hidden_size, n_epochs))
    if plot_gradient:
        gradient_norms = []
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    for epoch in range(n_epochs):
        begin = time.time()
        losses = []

        #for i in range(num_examples // batch_size):
        optimizer.zero_grad()
        
        random_batch_idx = random.sample(range(num_examples), batch_size)
        #random_batch_idx = random.sample(range(num_examples), num_examples)
        batch = input[random_batch_idx]
        output = net(batch)
        if epoch==0:
            output0 = output
        loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
        
        losses.append(loss.item())
        all_losses.append(loss.item())
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        if plot_gradient:
            tot = 0
            for param in [p for p in net.parameters() if p.requires_grad]:
                tot += (param.grad ** 2).sum()
            gradient_norms.append(sqrt(tot))
        #This is for debugging
        # for param in [p for p in net.parameters() if p.requires_grad]:
        #     graD[:,epoch] = param.grad.detach().numpy()[:,0]
        optimizer.step()
        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()

        if np.mod(epoch, 40)==0 and verbose is True:
            if keep_best and np.mean(losses) < best_loss:
                best = net.clone()
                best_loss = np.mean(losses)
                print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
            else:
                print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if torch.cuda.is_available():
            bs[:,epoch]  = net.cpu().b.detach().numpy()[:,0]
            gs[:,epoch]  = net.cpu().g.detach().numpy()[:,0]
            wr[:,:,epoch] = net.cpu().wrec.detach().numpy()      
            net.to(device=device)
        else:
            bs[:,epoch]  = net.b.detach().numpy()[:,0]
            gs[:,epoch]  = net.g.detach().numpy()[:,0]
            wr[:,:,epoch] = net.wrec.detach().numpy()
        
            
    if plot_learning_curve:
        plt.figure()
        plt.plot(all_losses)
        plt.yscale('log')
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.figure()
        plt.plot(gradient_norms)
        plt.yscale('log')
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())
    if save_loss==True and save_params==True:
        return(all_losses, bs, gs, wr, output, output0)
    elif save_loss==False and save_params==True:
        return(bs, gs, wr, output, output0)
    elif save_loss==True and save_params==False:
        return(all_losses, output, output0)
    else:
        return()
    
def give_output(net, _input, cuda=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    net.to(device=device)
    input = _input.to(device=device)
    output = net(input)
    return(output)


#%%
### load full dataset
dataset = "hemibrain"
#dataset = "flywire"
datapath = "Data/Figure5/exported-traced-adjacencies-v1.2/"
if dataset == "hemibrain":
    resort = False

    neuronsall = pd.read_csv(datapath+"traced-neurons.csv")
    neuronsall.sort_values(by=['instance'],ignore_index=True,inplace=True)
    conns = pd.read_csv(datapath+"traced-total-connections.csv")

    Nall = len(neuronsall)
    Jall = np.zeros([Nall,Nall],dtype=np.uint)

    idhash = dict(zip(neuronsall.bodyId,np.arange(Nall)))
    preinds = [idhash[x] for x in conns.bodyId_pre]
    postinds = [idhash[x] for x in conns.bodyId_post]

    Jall[postinds,preinds] = conns.weight


### identify cell types 
types = np.array(neuronsall.type).astype(str)
def getsubtype(t,types,resort=False):
    inds = np.nonzero([t in x for x in types])[0]
    if resort:
        sortinds = np.argsort(types[inds])
        inds = inds[sortinds]
    return inds

types = np.array(neuronsall.type).astype(str)
#%%
#key ring attractor cell types
epg = getsubtype("EPG",types,resort=resort)
pen = getsubtype("PEN",types,resort=resort)
peg = getsubtype("PEG",types,resort=resort)
delta7 = getsubtype("Delta7",types,resort=resort)


allcx = np.concatenate((epg,pen,delta7,peg)) #neurons to keep

allcx[0:46]=allcx[[23,24, 0,1, 42,43,44,45, 2,3, 39,40,41,  4,5,6,\
                   36,37,38,  7,8,9,   33,34,35,   10,11,12,  \
                    30,31,32,  13,14,15,  27,28,29, 16,17,18, \
                    25,26,  19,20,21,22]]

epg_ix = [0,0, 1,1, 2,2,2,2, 3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9, \
          10,10,10,  11,11,11,  12,12,12,  13,13,13,  14,14, 15,15,15,15 ]

#% subselect only the neurons in allcx
J = 1.*Jall[allcx,:][:,allcx]
if dataset == "flywire":
    J = np.array(J.todense())
N = J.shape[0]
neurons = neuronsall.iloc[allcx,:]
neurons.reset_index(inplace=True)

W_16to46 = np.zeros((46,16))
for i in range(len(np.unique(epg_ix))):
    ixx = np.where(np.array(epg_ix)==i)[0]
    W_16to46[ixx,i] = 1

W_46to16 = W_16to46/np.sum(W_16to46,0)

W_46to16 = W_46to16.T

W_16to3 = np.zeros((16,3))
for i in range(len(np.unique(epg_ix))):
    ori = (i/16)*2*np.pi-np.pi
    W_16to3[i,0] = np.cos(ori)
    W_16to3[i,1] = np.sin(ori)

W_46to3 = W_16to3.T.dot(W_46to16)
W_46to3[2,:] = 1./46
    
#%%
uniqtypes = pd.unique(neurons.type)
Ntype = len(uniqtypes)
typehash = dict(zip(uniqtypes,np.arange(Ntype)))
typeclasses = np.array([typehash[x] for x in neurons.type]) #neuron i belongs to cell type uniqtypes[typeclasses[i]]
#%%
typeinds = [np.where(neurons.type == uniqtypes[ii])[0] for ii in range(Ntype)] #typeinds[i] are the indices of neurons belonging to cell type uniqtypes[i]
Npertype = np.array([len(x) for x in typeinds]) #Npertype[i] is the number of neurons of cell type uniqtypes[i]

types_1hot = np.zeros([N,Ntype])
types_1hot[np.arange(N),typeclasses] = 1.


#%%
# Normalization of connectivity
J2 = np.copy(J)
J2[:,types_1hot[:,-2]==1.] = -5*np.abs(J[:,types_1hot[:,-2]==1.])
u = np.linalg.eigvals(J2)

Jf = 0.9*J2/np.max(np.real(u))
u, v = np.linalg.eig(Jf)
ax = set_axis()
plt.scatter(np.real(u), np.imag(u))
plt.xlabel(r'real $\lambda$')
plt.ylabel(r'imag $\lambda$')
plt.show()

#%%


Trials = 50
TT = 10
Factor = 0.1
FactorInp = 5.
#np.random.seed(14)
y,m,inps, ts = generate_targets(Trials, epg_ix)
angls = get_angles(y)
Ttrr = 15
fig = plt.figure(figsize=[4,4])
ax = fig.add_subplot(211)
plt.plot(ts, angls[Ttrr,:])
plt.ylim([-np.pi, np.pi])
plt.ylabel('angle (rad.)')
# plt.plot(ts, y[Ttrr,:,0])
# plt.plot(ts, np.sqrt(y[Ttrr,:,0]**2+y[Ttrr,:,1]**2))

ax = fig.add_subplot(212)
plt.plot(ts, -inps[Ttrr,:,-1], label='PEN1 Left')
plt.plot(ts, inps[Ttrr,:,-2], label = '- PEN1 Right')
plt.legend()
plt.xlabel('time (tau)')
plt.ylabel('input')
plt.show()

plt.imshow(inps[Ttrr, :, 0:46].T, vmin=0, vmax=5.)
plt.show()
#%%
J2 = np.copy(J)
J2[:,types_1hot[:,-2]==1.] = -2*np.abs(J[:,types_1hot[:,-2]==1.])
u, v = np.linalg.eig(J2)
Jf = 0.9*J2/np.max(np.real(u))
N = np.shape(Jf)[0]
input_size = 46+2
bt = 0.1*np.random.randn(N)
gt = 0.1*np.random.randn(N)-0.2
x0 = 0.05*np.random.randn(N)-0.1
Max = 5.
Min = 0.2
taus0 = np.arctanh((2-(Max+Min))/(Max-Min))*np.ones(N)
output_size = 3+46
wout = np.zeros((N,output_size))
wout[0:46,0:3] = W_46to3.T
wout[0:46, 3:] = np.eye(46)
winp = np.zeros((input_size,N))
sinp = np.zeros((input_size,1))
for ii in range(46):
    winp[ii,ii] = 2.
winp[-1,50:60] = 1. #this are the left neurons. Activating them should move bump backwards
winp[-2,60:70] = 1.

dtype = torch.FloatTensor 
Wr_ini =Jf
Wr_ini_log = np.copy(Jf)
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

Net = RNN(input_size, hidden_size, output_size, wi_init, si_init, wo_init, wrec_init, mwrec_init, b_init, g_init, taus_init,
          h0_init=h0_init, train_b = True, train_g=True, train_wi = False, train_wout=False, train_conn = False, noise_std=0.05,
          alpha=alpha, linear=lin_var, train_h0=True, beta=Beta, train_taus=False)
Wbump=1.0
mValue = 0.5
do_times = 1
n_ep = 1500 #3000 for first step
lr= 4e-3

Trials = 150#40
TStay = 3.
TT = 11.
losses = []

trainstr = '' #''
#%%
y2,m2,inps2, ts2 = generate_targets(Trials,  epg_ix, wbump=Wbump)
#y,m,inps, ts = generate_targets(Trials, epg_ix)
dt = ts2[1]-ts2[0]
m2[:,:,2] = mValue*m2[:,:,2]
y = y2[:,0:int((TT-TT//3)//dt),:]
m = m2[:,0:int((TT-TT//3)//dt),:]
inps = inps2[:,0:int((TT-TT//3)//dt),:]
ts = ts2[0:int((TT-TT//3)//dt)]

input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(y)
mask_train = torch.from_numpy(m).type(dtype)
outStu = Net.forward(input_Train)
out = outStu.detach().numpy()

plt.plot(out[0,:,0], c='k')
plt.plot(out[0,:,1], c='k')
plt.plot(out[0,:,2], c='k')
plt.plot(out[0,:,3:])
#%%
plt.plot(inps2[1,:,:])
y,m,inps, ts = generate_targets(Trials, epg_ix)
plt.plot(inps[1,:,:])


#%%

print('number of trials: '+str(Trials))

JJ =np.exp(Net.wrec.detach().numpy())*np.sign(Net.mwrec.detach().numpy())
gg = np.exp(Net.g.detach().numpy()[:,0])
bb = Net.b.detach().numpy()[:,0]
hh0 = Net.h0.detach().numpy()
wI = Net.wi.detach().numpy()
wOut = Net.wout.detach().numpy()
alpha_ = Net.alpha
si_ = Net.si.detach().numpy()

np.savez('params_netSimpleRing2_init.npz', JJ, gg, bb, hh0, wI, wOut, alpha_, si_)



y2,m2,inps2, ts2 = generate_targets(Trials,  epg_ix, T=TT, wbump=Wbump)
dt = ts2[1]-ts2[0]
m2[:,:,2] = mValue*m2[:,:,2]
y = y2[:,0:int((TT-TT//3)//dt),:]
m = m2[:,0:int((TT-TT//3)//dt),:]
inps = inps2[:,0:int((TT-TT//3)//dt),:]
ts = ts2[0:int((TT-TT//3)//dt)]

input_Train = torch.from_numpy(inps).type(dtype)
out_Train = torch.from_numpy(y)
mask_train = torch.from_numpy(m).type(dtype)
outStu = Net.forward(input_Train)

print('training short')
lossJ__, bsJ, gsJ__, wr, outJ, outJ_ = train(Net, input_Train, out_Train, mask_train, n_ep, hidden_size,  plot_learning_curve=False, plot_gradient=False, 
                                  lr=lr, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True, batch_size=np.min((30,Trials)))


losses.append(lossJ__)
outStu = Net.forward(input_Train)
out = outStu.detach().numpy()


torch.save(Net.state_dict(), 'netPopVec_Wrec_simplering2.pt')

np.savez('loss_a_simplering2.npz', lossJ__)

JJ =np.exp(Net.wrec.detach().numpy())*np.sign(Net.mwrec.detach().numpy())
gg = np.exp(Net.g.detach().numpy()[:,0])
bb = Net.b.detach().numpy()[:,0]
hh0 = Net.h0.detach().numpy()
wI = Net.wi.detach().numpy()
wOut = Net.wout.detach().numpy()
alpha_ = Net.alpha
si_ = Net.si.detach().numpy()
np.savez('params_netSimpleRing2_final.npz', JJ, gg, bb, hh0, wI, wOut, alpha_, si_)

#%%
y3,m3,inps3, ts3 = generate_targets(Trials*2, epg_ix, wbump=Wbump, T=TT, )
dt = ts2[1]-ts2[0]
m3[:,:,2] = mValue*m3[:,:,2]

input_Train3 = torch.from_numpy(inps3).type(dtype)
out_Train3 = torch.from_numpy(y3)
mask_train3 = torch.from_numpy(m3).type(dtype)
outStu3 = Net.forward(input_Train3)
out3 = outStu3.detach().numpy()

Aout = get_angles(out3)
Atar = get_angles(y3)

ax = set_axis()
Trs = np.random.randint(0, Trials, size=2)
for iT in range(len(Trs)):
    plt.plot(ts3, Aout[Trs[iT],:].T, c='C'+str(iT))
    plt.plot(ts3, Atar[Trs[iT],:].T, '--', c='C'+str(iT))
plt.ylim([-1.1*np.pi, 1.1*np.pi])
plt.xlabel('time')
 
plt.ylabel('decoded angle')
plt.savefig('Figs/simplering2_netAngle_final.pdf')
plt.show() 

fig = plt.figure(figsize=[7,3])
ax = set_axis(fig, 131)
for iT in range(len(Trs)):
    plt.plot(ts3, out3[Trs[iT],:,0].T, c='C'+str(iT))
    plt.plot(ts3, y3[Trs[iT],:,0].T, '--', c='C'+str(iT))
    plt.plot(ts3, m3[Trs[iT],:,0], c='k')
ax.set_xlabel('time')
ax.set_ylabel('proj x')

ax = set_axis(fig, 132)
for iT in range(len(Trs)):
    plt.plot(ts3, out3[Trs[iT],:,1].T, c='C'+str(iT))
    plt.plot(ts3, y3[Trs[iT],:,1].T, '--', c='C'+str(iT))
    plt.plot(ts3, m3[Trs[iT],:,1], c='k')
#plt.ylim([-1.1*np.pi, 1.1*np.pi])
ax.set_xlabel('time')
plt.ylabel('proj y')

ax = set_axis(fig, 133)
for iT in range(len(Trs)):
    plt.plot(ts3, out3[Trs[iT],:,2].T, c='C'+str(iT))
    plt.plot(ts3, y3[Trs[iT],:,2].T, '--', c='C'+str(iT))
    plt.plot(ts3, m3[Trs[iT],:,2], c='k')
#plt.ylim([-1.1*np.pi, 1.1*np.pi])
ax.set_xlabel('time')
plt.ylabel('rate')
plt.savefig('Figs/simplering2_netAngle_readout_final.pdf')
plt.show() 
#%%
for iT in range(len(Trs)):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.imshow(out3[Trs[iT],:,3:].T, aspect='auto', vmax = 0.8*np.max(out[Trs[iT],:,3:].T))
    plt.colorbar()
    ax = fig.add_subplot(122)
    plt.imshow(y3[Trs[iT],:,3:].T, aspect='auto', vmax = 0.9*np.max(y[Trs[iT],:,3:].T))
    plt.colorbar()
    plt.savefig('Figs/simplering2_netAngle_exampleBump_trial_'+str(iT)+'.pdf')
    plt.show()
#%%
plt.figure()
plt.scatter(np.ravel(Atar[:, 20:]),np.ravel(Aout[:, 20:]), alpha=.2, rasterized=True)
plt.xlabel('target')
plt.ylabel('produced')
plt.savefig('Figs/simplering2_netAngle_exampleBump_trialFinal_'+str(iT)+'.pdf', dpi=200)
plt.show()
#%%
a=(np.abs(Atar- Aout))
iBad = np.unravel_index(a.argmax(), a.shape)
#%%
ii = -1
plt.plot(Atar[iBad[0],:])
plt.plot(Aout[iBad[0],:])
#%%
plt.imshow(out3[iBad[ii],:,3:].T, vmin=0, vmax=5)
#%%
plt.plot(out3[iBad[ii],:,0:3])
Aout_ = get_angles(out3[iBad[ii]:iBad[ii]+1,:,:])

plt.show()
plt.plot(Aout_)
plt.plot(Aout[iBad[ii],:])
plt.plot(Atar[iBad[ii],:],'--', c='k')
plt.show()
#%%
fig = plt.figure(figsize=[6,4])
ax = fig.add_subplot(131)
Jfin = np.exp(Net.wrec.detach().numpy())*np.sign(Net.mwrec.detach().numpy()).dot(np.diag(np.exp(Net.g.detach().numpy()[:,0])))
plt.imshow(Jfin,  cmap = 'RdBu', vmin=-0.1, vmax=0.1)
#plt.colorbar()
plt.xlabel('pre-synaptic')
plt.ylabel('post-synaptic')
ax.set_title('final', fontsize=13)
ax = fig.add_subplot(132)
Jfin0 = np.exp(wrec_init.detach().numpy())*np.sign(Net.mwrec.detach().numpy()).dot(np.diag(np.exp(g_init.detach().numpy()[:,0])))
plt.imshow(Jfin0,  cmap = 'RdBu', vmin=-0.1, vmax=0.1)
ax.set_title('initial', fontsize=13)
#plt.colorbar()
plt.xlabel('pre-synaptic')
plt.ylabel('post-synaptic')

ax = fig.add_subplot(133)
Jfin = np.exp(Net.wrec.detach().numpy())*np.sign(Net.mwrec.detach().numpy()).dot(np.diag(np.exp(Net.g.detach().numpy()[:,0])))
Jfin0 = np.exp(wrec_init.detach().numpy())*np.sign(Net.mwrec.detach().numpy()).dot(np.diag(np.exp(g_init.detach().numpy()[:,0])))
plt.imshow(Jfin-Jfin0,  cmap = 'PRGn', vmin=-0.2, vmax=0.2)
#plt.colorbar()
plt.xlabel('pre-synaptic')
plt.ylabel('post-synaptic')
ax.set_title('diff', fontsize=13)

plt.savefig('Figs/simplering2_matrix_final.pdf')
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ufin = np.linalg.eigvals(Jfin)
ufin0 = np.linalg.eigvals(Jfin0)
plt.scatter(np.real(ufin), np.imag(ufin), s=30, c='white', edgecolor='k', label='trained')
plt.scatter(np.real(ufin0), np.imag(ufin0), s=30, c='k',  label='initial')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'Re($\lambda$)')
ax.set_ylabel(r'Im($\lambda$)')
plt.legend()
plt.savefig('Figs/simplering2_matrix_spect.pdf')
#%%
fig = plt.figure(figsize=[5,3])
ax = set_axis(fig,121)
xx = np.ravel(Jfin0[Jfin0!=0.])
yy = np.ravel(Jfin[Jfin!=0.])
plt.scatter(xx[xx>0], yy[yy>0], label='exc', alpha=0.7, c='C1', rasterized=True)
plt.yscale('log')
plt.xscale('log')
plt.plot([1e-3, 1], [1e-3, 1], c='k')
plt.legend()
ax.set_xlabel('initial weight')
ax.set_ylabel('trained weight')

ax = set_axis(fig,122)
xx = np.ravel(Jfin0[Jfin0!=0.])
yy = np.ravel(Jfin[Jfin!=0.])
plt.scatter(-xx[-xx>0], -yy[-yy>0], label='inh', alpha=0.7, rasterized=True)
plt.yscale('log')
plt.xscale('log')
plt.plot([1e-3, 1], [1e-3, 1], c='k')
plt.legend()
ax.set_xlabel('initial weight')
ax.set_ylabel('trained weight')
plt.savefig('Figs/simplering2_matrix_scatter.pdf', dpi=200)

plt.savefig('Figs/simplering2_synws_hist.pdf',dpi=200)
#%%    
Jfin = np.exp(Net.wrec.detach().numpy())*np.sign(Net.mwrec.detach().numpy())#.dot(np.diag(np.exp(Net.g.detach().numpy()[:,0])))
Jfin0 = np.exp(wrec_init.detach().numpy())*np.sign(Net.mwrec.detach().numpy())#.dot(np.diag(np.exp(g_init.detach().numpy()[:,0])))
gfin = Net.g.detach().numpy()
g0 = g_init.detach().numpy()
np.savez('initial_simplering2_mat.npz', Jfin, Jfin0, gfin, g0) 
#%%
JJ =np.exp(Net.wrec.detach().numpy())*np.sign(Net.mwrec.detach().numpy())
gg = np.exp(Net.g.detach().numpy()[:,0])
bb = Net.b.detach().numpy()[:,0]
hh0 = Net.h0.detach().numpy()

Ntr = 300
Ts = np.arange(0,20,dt)
trajs_h = np.zeros((len(Ts),N,Ntr))
trajs_r = np.zeros((len(Ts),N,Ntr))
dt = Ts[1]-Ts[0]

for itr in range(Ntr):
    if itr<Ntr-1:
        h0 = 3*np.random.randn()#+hh0
    else:
        h0 = hh0
    r = gg*softplus(h0+bb, beta=Beta)
    trajs_h[0,:,itr] = h0
    trajs_r[0,:,itr] = r
    for it, ti in enumerate(Ts[:-1]):
        h0 = h0 + dt*(-h0+JJ.dot(r))+np.sqrt(dt)*np.random.randn(N)*0.2
        r = gg*softplus(h0+bb, beta=Beta)
        trajs_h[it+1,:,itr] = h0
        trajs_r[it+1,:,itr] = r

dat = trajs_r[100:,:,:][:,:,np.isnan(trajs_r).sum(0).sum(0)==0]
dat0 = trajs_r[50:,:,:][:,:,np.isnan(trajs_r).sum(0).sum(0)==0]

X = dat-np.nanmean(np.nanmean(dat,0, keepdims=True),-1,keepdims=True)

C = 0
for i in range(np.shape(X)[0]):
    C += X[i,:,:].dot(X[i,:,:].T)/np.shape(X)[0]
u, v= np.linalg.eigh(C)
u = u[::-1]
v=v[:,::-1]

Proj = np.zeros_like(dat)
for i in range(np.shape(X)[0]):
    Proj[i,:,:] = v.T.dot(dat0[i,:,:])
#%%
ax = plt.figure().add_subplot(projection='3d')
for i in range(np.shape(Proj)[-1]):
    ax.plot(Proj[:,0,i], Proj[:,1,i], Proj[:,2,i] )
    ax.scatter(Proj[-1,0,i], Proj[-1,1,i], Proj[-1,2,i], c='k' )
    #ax.scatter(Proj[1,0,i], Proj[1,1,i], Proj[1,2,i], c='C0' )
    
ax.plot(Proj[:,0,i], Proj[:,1,i], Proj[:,2,i] , c='k', lw=6)
ax.plot(Proj[:,0,i], Proj[:,1,i], Proj[:,2,i] , c=[1., 1., 0.5], lw=5)
plt.show()

#%%
plt.figure()
#plt.plot(trajs_r[-1,0:46,:])
plt.plot(trajs_r[-1,:,-1], c='k', lw=3)
plt.plot(trajs_r[-1,:,0], c='C0', lw=3)

plt.show()
