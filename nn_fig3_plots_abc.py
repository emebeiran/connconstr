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
import time
import random
from math import sqrt
from argparse import ArgumentParser
    #%%
stri = 'Data/Figure3/' 
plt.style.use('ggplot')

fig_width = 1.5*2.2 # width in inches
fig_height = 1.5*2  # height in inches
fig_size =  [fig_width,fig_height]
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['figure.autolayout'] = True
 
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['lines.markeredgewidth'] = 0.003
plt.rcParams['lines.markersize'] = 3
plt.rcParams['font.size'] = 14#9
plt.rcParams['legend.fontsize'] = 11#7.
plt.rcParams['axes.facecolor'] = '1'
plt.rcParams['axes.edgecolor'] = '0'
plt.rcParams['axes.linewidth'] = '0.7'

plt.rcParams['axes.labelcolor'] = '0'
plt.rcParams['axes.labelsize'] = 14#9
plt.rcParams['xtick.labelsize'] = 11#7
plt.rcParams['ytick.labelsize'] = 11#7
plt.rcParams['xtick.color'] = '0'
plt.rcParams['ytick.color'] = '0'
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['axes.grid'] = False
plt.rcParams['font.sans-serif'] = 'Arial'

#%%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, b_init, g_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_wout = True, train_wi = False, train_h0=False, noise_std=0., alpha=0.2, linear=False):
        
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
            self.non_linearity = torch.tanh
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if train_wi:
            self.wi.requires_grad= True
        else:
            self.wi.requires_grad= False
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False
        else:
            self.h0.requires_grad = True
        
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
            self.wrec.copy_(wrec_init)
            self.wout.copy_(wo_init)
            self.b.copy_(b_init)
            self.g.copy_(g_init)
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
        output[:,0,:] = torch.mul(self.g.T, r).matmul(self.wout)
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(self.g.T, r).matmul(self.wrec.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h+self.b.T)
            output[:,i+1,:] = torch.mul(self.g.T, r).matmul(self.wout)
        
        return output
    
#%%
class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, b_init, g_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_wout = True, noise_std=0., alpha=0.2, linear=False):
        
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
            self.non_linearity = torch.tanh
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.wi.requires_grad= False
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        self.h0.requires_grad = False
        
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
            self.wrec.copy_(wrec_init)
            self.wout.copy_(wo_init)
            self.b.copy_(b_init)
            self.g.copy_(g_init)
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
        output[:,0,:] = torch.mul(self.g.T, r).matmul(self.wout)
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(self.g.T, r).matmul(self.wrec.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h+self.b.T)
            output[:,i+1,:] = torch.mul(self.g.T, r).matmul(self.wout)
        
        return output
    
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
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


#%%
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

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
    return(initial_loss.item())

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
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

        if np.mod(epoch, 10)==0 and verbose is True:
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
    num_examples = _input.shape[0]
    
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
# =============================================================================
#   Generate network
# =============================================================================
N = 200
sigma_g = 0.9

Sigma_mn = np.zeros((4,4))
Sigma_mn[0,0] = 1
Sigma_mn[1,1] = 1
Sigma_mn[2,2] = 1
Sigma_mn[3,3] = 1
Sigma_mn[0,2] = 1.5
Sigma_mn[2,0] = Sigma_mn[0,2]
Sigma_mn[0,3] = 3.
Sigma_mn[3,0] = Sigma_mn[0,3]
Sigma_mn[1,2] = -3.
Sigma_mn[2,1] = Sigma_mn[1,2]
Sigma_mn[1,3] = 1.5
Sigma_mn[3,1] = Sigma_mn[1,3]

Mu = np.zeros((4,))

max_iter = 100
for it in range(max_iter):
    vals = np.linalg.eigvals(Sigma_mn)
    if np.min(vals)<0:
        Sigma_mn[2,2] = 1.1*Sigma_mn[2,2]
        Sigma_mn[3,3] = 1.1*Sigma_mn[3,3]
    else:
        Sigma_mn[2,2] = 1.1*Sigma_mn[2,2]
        Sigma_mn[3,3] = 1.1*Sigma_mn[3,3]
        break
    
X = np.random.multivariate_normal( Mu, Sigma_mn, N)
m1 = X[:,0][:,np.newaxis]
m2 = X[:,1][:,np.newaxis]
n1 = X[:,2][:,np.newaxis]
n2 = X[:,3][:,np.newaxis]

J = (m1.dot(n1.T) + m2.dot(n2.T))/N 

g = sigma_g*np.random.randn(N,1)+1

Jef = J.dot(np.diag(g[:,0]))

U, S, V = np.linalg.svd(Jef)
#wout = U[:,0]/np.sqrt(N)

dta = 0.05
taus = np.arange(0, 20, dta)
Nt = len(taus)
Tau = 0.5

ics = np.arange(8)
count = 0

output_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 64]#, 64, 100]#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32]
lrs = [0.005,]#[0.001, 0.01, 0.1, 0.5, 1., 5.]
Ns = [100, 200, 400, 600, 800, 1000]#, 1200]#, 1200]#, 1400, 1600]#[100, 200, 400, 600, 800, 1000, 1200, 1400, 1600]

lr = 0.005
#%%
try:
    sA = np.load(stri+'LRFinalnet_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')#np.load('LRFinalnet_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')#np.load('LRGCVnet_sevN2_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')
    # Problem with last N
    losses = sA['arr_0']#[:,:,:-1,:]
    egm = sA['arr_1']#[:,:,:-1,:]
    egm2 = sA['arr_2']#[:,:,:-1,:]
    #E_Ts = sA['arr_3'][:,:,:-1,:]
    E_knos = sA['arr_4']#[:,:-1,:]
    E_unks = sA['arr_5']#[:,:-1,:]
    
    print('loaded')
except: 
    for iN, N in enumerate(Ns):
        print('N='+str(N))
        print('--')
        for ip, output_size in enumerate(output_sizes):
          
            print(output_size)
            
            wout = np.zeros((N,output_size))
            for ii in range(output_size):
                wout[ii,ii]  = 1
    
            input_size = 2
            hidden_size = N
            trials = 20
            
            x_init = np.random.randn(N)#x0
            x0 = np.copy(x_init)
            seed =21
            sig_b = 0.
            
            bt = sig_b*(2*np.random.rand(N)-1.)
            gt = 1+np.random.randn(N,1)
            
            dtype = torch.FloatTensor  
            wi_init = torch.from_numpy(x_init).type(dtype)
            wo_init = torch.from_numpy(wout).type(dtype)
            
            output_size2 = N
            wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
            
            
            
            factor = 0.00001
            Wr_ini =np.random.randn(N,N)/np.sqrt(N)
            wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
            b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
            g_init_Tru = torch.from_numpy(gt).type(dtype)
            b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
            g_init_Ran = torch.from_numpy(factor*sigma_g*np.random.randn(N,1)+1).type(dtype)
            #g_init_Ran = torch.from_numpy(sigma_g*np.random.randn(N,1)+1).type(dtype)
            alpha = dta/Tau
            input_train = np.zeros((trials, Nt, input_size))
            input_Train = torch.from_numpy(input_train).type(dtype)
            
            x0s = np.random.randn(N)
            h0_init = torch.from_numpy(x0).type(dtype)
            
            
            lin_var = False
            #NetJ_Teacher = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
            #                                  train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
            #                                  alpha=alpha, linear=lin_var, train_h0=False)
                    
            NetJ         = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
                                              train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                              alpha=alpha, linear=lin_var, train_h0=False)
            
            NetJ_T_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                              train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                              alpha=alpha, linear=lin_var, train_h0=False)
            
            NetJ_0_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
                                              train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                              alpha=alpha, linear=lin_var, train_h0=False)
       
            for ilr, lr in enumerate(lrs):
                for ic in ics:
                    count_ic = 0
                    try:
                  
                        NetJ.load_state_dict(torch.load(stri+"LRGnet_J_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                        NetJ_T_all.load_state_dict(torch.load(stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                        NetJ_0_all.load_state_dict(torch.load(stri+"LRGnet_J0all_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                    
                        if count_ic==0 and ilr==0:
                            NetJ_all         = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, NetJ.wrec, NetJ.b, NetJ.g, h0_init=NetJ.h0,
                                                              train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                              alpha=alpha, linear=lin_var, train_h0=False)
                        #stri+'taskLR_losses_percSGD_N_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz'
                        A = np.load(stri+'taskLR_Glosses_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
                        loss = A['arr_0']
                        eg = A['arr_1']
                        gT = NetJ_T_all.g.detach().numpy()[:,0]
                        gS = NetJ.g.detach().numpy()[:,0]
                        g0 = NetJ_0_all.g.detach().numpy()[:,0]
                        
                        if count_ic ==0:
                            outTeacher = NetJ_T_all.forward(input_Train)
                            outteacher = outTeacher.detach().numpy()
                        
                        outStu = NetJ_all.forward(input_Train)
                        outstu = outStu.detach().numpy()
                        
                      
                        outstuN= outstu.reshape(-1, outstu.shape[-1])
                        outteacherN= outteacher.reshape(-1, outteacher.shape[-1])
                        CC = np.corrcoef(outstuN.T, outteacherN.T)
                        # CVs = np.zeros(N)
                        CVs = np.diag(CC[0:N,N:])

                        #E = (outstu[:,:,:].T-outteacher[:,:,:].T)**2
                        #Den = (outteacher[:,:,:].T-np.mean(outteacher.T,(1,2), keepdims=True))**2
                        #E2 = E
                        #E_N = np.sqrt(np.mean(np.mean(E,1),1)
                        #E_T = np.sqrt(np.mean(np.mean(E,0),1))
                        E_kno = 1-np.mean(np.abs(CVs[0:output_size]))
                        E_unk = 1-np.mean(np.abs(CVs[output_size:]))

                    except:
                        print(stri+"LRGRnet_J_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt')
                        print('file missing')
                        print('')
                    
                    
                    if count==0:
                        losses = np.zeros((len(loss), len(output_sizes), len(Ns), len(ics)))
                        egm = np.zeros((np.shape(eg)[1], len(output_sizes), len(Ns), len(ics)))
                        Dgm = np.zeros((np.shape(eg)[1], len(output_sizes), len(Ns), len(ics)))
                        #gFs = np.zeros((np.shape(eg)[0], len(output_sizes), len(Ns), len(ics)))
                        
                        egm2 = np.zeros((2, len(output_sizes), len(Ns), len(ics)))
                        #E_Ns = np.zeros((N, len(output_sizes), len(Ns), len(ics)))
                        E_Ts = np.zeros((len(taus), len(output_sizes), len(Ns), len(ics)))
                        E_knos = np.zeros(( len(output_sizes), len(Ns), len(ics)))
                        E_unks = np.zeros(( len(output_sizes), len(Ns), len(ics)))
                        count = count+1
                        
                        
                    try:
                        losses[:,ip, iN, ic] = loss
                        egm[:,ip, iN, ic] = np.sqrt(np.mean(eg**2,0))
                        #gFs[:,ip, iN, ic] = gS
                        egm2[1,ip, iN, ic] = np.sqrt(np.mean((gT-gS)**2))
                        egm2[0,ip, iN, ic] = np.sqrt(np.mean((gT-g0)**2))
                        #E_Ns[:,ip,iN,ic] = E_N
                        E_Ts[:,ip,iN,ic] = np.nan#E_T
                        E_knos[ip,iN,ic] = E_kno
                        E_unks[ip,iN,ic] = E_unk
                            
                        
                    except:
                        print('-- '+str(iN) + '  '+str(ic)+'  '+str(ip))
                        print('')
                        losses[:,ip, iN, ic] = np.nan
                        egm[:,ip, iN, ic] = np.nan
                        egm2[1,ip, iN, ic] = np.nan
                        egm2[0,ip, iN, ic] = np.nan
                    
                    count_ic = count_ic+1
    #the two calculates the egms differently
    print(np.shape(losses))
    np.savez('LRFinalnet_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz', losses, egm, egm2, E_Ts, E_knos, E_unks)

#%%


# Calculate zero value

#%%
try:
    bb = np.load(stri+'zero_values_lowrank.npz')
    E0s = bb['arr_0']
except:
    E0s = np.zeros((len(Ns), len(ics)))
    for iN, N in enumerate(Ns):
        print('N0='+str(N))
        print('--')
        ip = 0
        output_size = output_sizes[ip]
        
        wout = np.zeros((N,output_size))
        for ii in range(output_size):
            wout[ii,ii]  = 1
    
        input_size = 2
        hidden_size = N
        trials = 20
        
        x_init = np.random.randn(N)#x0
        x0 = np.copy(x_init)
        seed =21
        sig_b = 0.
        
        bt = sig_b*(2*np.random.rand(N)-1.)
        gt = 1+np.random.randn(N,1)
        
        dtype = torch.FloatTensor  
        wi_init = torch.from_numpy(x_init).type(dtype)
        wo_init = torch.from_numpy(wout).type(dtype)
        
        output_size2 = N
        wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
        
        factor = 0.00001
        Wr_ini =np.random.randn(N,N)/np.sqrt(N)
        wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
        b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
        g_init_Tru = torch.from_numpy(gt).type(dtype)
        b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
        g_init_Ran = torch.from_numpy(factor*sigma_g*np.random.randn(N,1)+1).type(dtype)
        #g_init_Ran = torch.from_numpy(sigma_g*np.random.randn(N,1)+1).type(dtype)
        alpha = dta/Tau
        input_train = np.zeros((trials, Nt, input_size))
        input_Train = torch.from_numpy(input_train).type(dtype)
        
        lin_var = False
    
        x0s = np.random.randn(N)
        h0_init = torch.from_numpy(x0).type(dtype)
        
        NetJ_0_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Ran, 
                                          train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                          alpha=alpha, linear=lin_var, train_h0=False)
        
        NetJ         = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
                                          train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                          alpha=alpha, linear=lin_var, train_h0=False)
        NetJ.load_state_dict(torch.load(stri+"LRGnet_J_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(0)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
        
        NetJ_all         = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, NetJ.wrec, NetJ.b, NetJ.g, h0_init=NetJ.h0,
                                                                  train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                                  alpha=alpha, linear=lin_var, train_h0=False)
        NetJ_T_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                          train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                          alpha=alpha, linear=lin_var, train_h0=False)
        
        NetJ_T_all.load_state_dict(torch.load(stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(0)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
        for ilr, lr in enumerate(lrs):
            for ic in ics:
                count_ic = 0
            
                NetJ_0_all.load_state_dict(torch.load(stri+"LRGnet_J0all_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                g0 = NetJ_0_all.g.detach().numpy()[:,0]
                
                if count_ic ==0:
                    outTeacher = NetJ_0_all.forward(input_Train)
                    outteacher = outTeacher.detach().numpy()
                
                outStu = NetJ_all.forward(input_Train)
                outstu = outStu.detach().numpy()
                
                outstuN= outstu.reshape(-1, outstu.shape[-1])
                outteacherN= outteacher.reshape(-1, outteacher.shape[-1])
                CC = np.corrcoef(outstuN.T, outteacherN.T)
                # CVs = np.zeros(N)
                CVs = np.diag(CC[0:N,N:])
    
                E0_kno = 1-np.mean(np.abs(CVs))
                E0s[iN, ic] = E0_kno
    np.savez(stri+'zero_values_lowrank.npz', E0s)
            
#%%
try:
    bb = np.load(stri+'refs_values_lowrank.npz')
    E_refs = bb['arr_0']
    E_refgs = bb['arr_1']
    
except:
    pers = 20
    E_refs = np.zeros((len(Ns), pers))
    E_refgs = np.zeros((len(Ns), pers))
    
    for iN, N in enumerate(Ns):
        print(N)
        ip = 0
        output_size= output_sizes[0]
    
        wout = np.zeros((N,output_size))
        for ii in range(output_size):
            wout[ii,ii]  = 1
        
        input_size = 2
        hidden_size = N
        trials = 20
        
        x_init = np.random.randn(N)#x0
        x0 = np.copy(x_init)
        seed =21
        sig_b = 0.
        
        bt = sig_b*(2*np.random.rand(N)-1.)
        gt = 1+np.random.randn(N,1)
        
        dtype = torch.FloatTensor  
        wi_init = torch.from_numpy(x_init).type(dtype)
        wo_init = torch.from_numpy(wout).type(dtype)
        
        output_size2 = N
        wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
        
        
        
        factor = 0.00001
        Wr_ini =np.random.randn(N,N)/np.sqrt(N)
        wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
        b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
        g_init_Tru = torch.from_numpy(gt).type(dtype)
        b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
        g_init_Ran = torch.from_numpy(factor*sigma_g*np.random.randn(N,1)+1).type(dtype)
        alpha = dta/Tau
        input_train = np.zeros((trials, Nt, input_size))
        input_Train = torch.from_numpy(input_train).type(dtype)
        
        x0s = np.random.randn(N)
        h0_init = torch.from_numpy(x0).type(dtype)
        
        
        lin_var = False
        NetJ_T_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                          train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                          alpha=alpha, linear=lin_var, train_h0=False)
    
           
        for ilr, lr in enumerate(lrs):
            ic=0
            NetJ_T_all.load_state_dict(torch.load(stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                    
        outTeacher = NetJ_T_all.forward(input_Train)
        outteacher = outTeacher.detach().numpy()
        outteacherN= outteacher.reshape(-1, outteacher.shape[-1])
        
        for ipe in range(pers):
            outstu = outteacher[:,:,np.random.permutation(N)]
            outstuN= outstu.reshape(-1, outstu.shape[-1])
            CC = np.corrcoef(outstuN.T, outteacherN.T)
            # CVs = np.zeros(N)
            CVs = np.diag(CC[0:N,N:])
        
            E_refs[iN, ipe] = 1-np.mean(np.abs(CVs[0:output_size]))
            E_refgs[iN, ipe] = np.sqrt(np.mean((gt[:,0]-gt[np.random.permutation(N), 0])**2))
        
    np.savez(stri+'refs_values_lowrank.npz', E_refs, E_refgs)       
#%%
palette2 = plt.cm.copper(np.linspace(0, 0.8, len(Ns)))

LW = 1
IC = 0
iN = 0
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)

for iN in np.arange(len(Ns)):
    if iN>-1:
        Y = np.copy(E_knos[:,iN,:])#/E_knos[:,iN,0][:,np.newaxis]
        print(np.sum(Y>0.001))
        Y[Y>0.001] = np.nan
        MY = np.nanmean(Y,-1)
        SY = np.nanstd(Y,-1)/np.sqrt(np.sum(~np.isnan(Y),-1))
        plt.plot(output_sizes, MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, c=palette2[iN], lw=2., label=str(Ns[iN]))    
        plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.5)
        # Y = E_unks[:,iN,:]#/E_unks[:,iN,0][:,np.newaxis]
        # MY = np.mean(Y,-1)
        # SY = np.std(Y,-1)/np.sqrt(len(ics))
        # plt.plot(output_sizes, MY, '--', c=palette2[iN], lw=2.)    
        # plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)


plt.legend(frameon=False, fontsize=10)
plt.xscale('log')
plt.yscale('log')

plt.ylim([0.00001, 0.01])
#plt.yticks([0.01,0.03])
#plt.yscale('log')

plt.ylabel(r'$1-|\rho|$ (recorded act.)')
plt.xlabel('rec. units $M$')
#plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
plt.savefig('Figs/Fig2_A_recorded.pdf', transparent=True)
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)

for iN in np.arange(len(Ns)):
    if iN>-1:
        Ym = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        mask = Ym>0.001
        print(np.sum(mask))
        # Y = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        # MY = np.mean(Y,-1)
        # SY = np.std(Y,-1)/np.sqrt(len(ics))
        # plt.plot(output_sizes, MY, c=palette2[iN], lw=2., label='N='+str(Ns[iN]))    
        # plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)
        Y = E_unks[:,iN,:]#/E_unks[:,iN,0][:,np.newaxis]
        Y[mask] = np.nan

        MY = np.nanmean(Y,-1)
        SY = np.nanstd(Y,-1)/np.sqrt(np.sum(~np.isnan(Y),-1))
        plt.plot(output_sizes, MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,  c=palette2[iN], lw=2.,  label=str(Ns[iN]))    
        plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.5)
        
        MY2 = np.nanmean(E0s[iN,:])
        SY2 = np.nanstd(E0s[iN,:])/np.sqrt(len(ics))
        print(MY2)
        plt.scatter(0.5, MY2, c=palette2[iN], s=25, edgecolor='k', linewidth=0.5,)
        plt.plot([0.5, 0.5], [MY2-SY2, MY2+SY2])
plt.legend(frameon=False, fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.00002, 0.8])
xl = ax.get_xlim()
#plt.plot(xl, np.ones_like(xl)*np.mean(E_refs), lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
plt.ylabel(r'$1-\left[|\rho|\right]$ (unrecorded act.)')
plt.xlabel('rec. units $M$')
#plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
plt.savefig('Figs/Fig2_A_unrecorded.pdf', transparent=True)

#%%
fig = plt.figure(figsize=[0.8*1.5*2.2*0.7, 0.8*1.5*2*0.7])
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)

for iN in np.arange(len(Ns)):
    if iN>-1:
        Ym = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        mask = Ym>0.001
        print(np.sum(mask))
        # Y = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        # MY = np.mean(Y,-1)
        # SY = np.std(Y,-1)/np.sqrt(len(ics))
        # plt.plot(output_sizes, MY, c=palette2[iN], lw=2., label='N='+str(Ns[iN]))    
        # plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)
        Y = E_unks[:,iN,:]#/E_unks[:,iN,0][:,np.newaxis]
        Y[mask] = np.nan

        MY = np.nanmean(Y,-1)
        SY = np.nanstd(Y,-1)/np.sqrt(np.sum(~np.isnan(Y),-1))
        plt.plot(output_sizes, MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,  c=palette2[iN], lw=2.,  label=str(Ns[iN]))    
        plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.5)
        
        MY2 = np.nanmean(E0s[iN,:])
        SY2 = np.nanstd(E0s[iN,:])/np.sqrt(len(ics))
        print(MY2)
        plt.scatter(0.5, MY2, c=palette2[iN], s=25, edgecolor='k', linewidth=0.5,)
        plt.plot([0.5, 0.5], [MY2-SY2, MY2+SY2])
#plt.legend(frameon=False, fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.00002, 0.004])
xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*np.mean(E_refs), lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
plt.ylabel(r'$1-|\rho|$ ')
plt.xlabel('rec. units $M$')
#plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
plt.savefig('Figs/Fig2_A_unrecorded_zoom.pdf', transparent=True)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)


for iN in np.arange(len(Ns)):
    if iN>0:
        Ym = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        mask = Ym>0.001
        print(np.sum(mask))
        Y = egm[-1,:,iN,:]#/egm[0,iPP,iN,:]
        Y[mask]=np.nan
        MY = np.nanmean(Y,-1)
        SY = np.nanstd(Y,-1)/np.sqrt(np.sum(~np.isnan(Y),-1))
        XX = np.arange(len(MY))
        plt.plot(output_sizes, MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, c=palette2[iN], lw=3., label='N='+str(Ns[iN]))    
        plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)
        
        Y0 = egm[0,0,iN,:]#/egm[0,iPP,iN,:]
        MY0 = np.nanmean(Y0)
        SY0 = np.nanmean(Y0)
        #plt.fill_between([0.5,], MY0-SY0, MY0+SY0, color=palette2[iN], alpha=0.2)
        plt.scatter(0.5, MY0, edgecolor='k', linewidth=0.5, c=palette2[iN],s=20)

#plt.legend()
plt.xscale('log')
xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*np.mean(E_refgs), lw=3, alpha=0.3, c='k')
ax.set_xlim(xl)
plt.ylim([0, 1.5])
plt.xticks([0.5, 1, 10, 100, ])
ax.set_xticklabels(['0', 1, 10, 100, ])
plt.ylabel(r'error gains')
plt.xlabel('rec. units $M$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/Fig2_A_gains.pdf')


#%%
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)

for iN in np.arange(len(Ns)):
    if iN>-1:
        Ym = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        mask = Ym>0.001
        print(np.sum(mask))
        # Y = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        # MY = np.mean(Y,-1)
        # SY = np.std(Y,-1)/np.sqrt(len(ics))
        # plt.plot(output_sizes, MY, c=palette2[iN], lw=2., label='N='+str(Ns[iN]))    
        # plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)
        Y = E_unks[:,iN,:]#/E_unks[:,iN,0][:,np.newaxis]
        Y[mask]=np.nan
        MY = np.nanmean(Y,-1)
        SY = np.nanstd(Y,-1)/np.sqrt(np.sum(~np.isnan(Y),-1))
        plt.plot(np.array(output_sizes)/Ns[iN], MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,  c=palette2[iN], lw=2.)    
        plt.fill_between(np.array(output_sizes)/Ns[iN], MY-SY, MY+SY, color=palette2[iN], alpha=0.2)

#plt.legend()
plt.xscale('log')

plt.xticks([0.001, 0.01, 0.1, 1.])

# plt.yscale('log')
# plt.yticks([0.01,0.03])

plt.ylabel(r'error unrecorded act.')
plt.xlabel('fraction rec. units')
#plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
plt.savefig('Figs/Fig2_A_unrecorded_fraction.pdf')

#%%
palette = plt.cm.plasma(np.linspace(0, 0.8, len(output_sizes)))
fig = plt.figure()
ax = fig.add_subplot(111)
iN0 = 0
for iPP in range(len(output_sizes)):
    if iPP>-1:

        mask = E_knos>0.0001#/E_knos[:,iN,0][:,np.newaxis]
        egm[-1,mask] = np.nan
        egm[0,mask] = np.nan

        plt.scatter(Ns[iN0:], (np.nanmean(egm[-1,iPP,iN0:,:]/egm[0,iPP,iN0:,:],-1)), s=20, color=palette[iPP])
            
        MY = 1*(np.nanmean(egm[-1,iPP,iN0:,:]/egm[0,iPP,iN0:,:],-1))
        SY = np.nanstd(egm[-1,iPP,iN0:,:]/egm[0,iPP,iN0:,:],-1)/np.sqrt(np.shape(egm[-1,iPP,iN0:,:])[-1])
        plt.plot(Ns[iN0:], MY, color=palette[iPP])
        plt.fill_between(np.array(Ns[iN0:]), MY-SY, MY+SY,  color=palette[iPP], alpha=.5)
        print(MY)
    
xx = np.linspace(100, 1000)
#plt.plot(xx, (1000/xx), '--', label=r'$1-2N^{-1}$', c='k')
#plt.legend()
#plt.yscale('log')
plt.xscale('log')
plt.xticks([ 100, 1000])
#plt.yticks([0.1, 1., 10., 100])
plt.ylabel(r'$\Delta$  error gains')
plt.xlabel(r'size $N$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/Fig2_A_changeGains_vs_N.pdf')

#%%
doPCAplot = True
if doPCAplot:
    try:
        Us=[]
        for iN, N in enumerate(Ns):
            aa = np.load('Data/Figure3/pc_abc_N_'+str(N)+'.npz')
            us = aa['arr_0']
            Us.append(us)           
            plt.ylabel('PC 2')
            plt.xlabel(r'PC 1')
            plt.plot([-1.0, 0.6],[-1.2, -1.2], c='k')
            plt.plot([-1., -1.],[-1, 1], c='k')
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.axis('off')
        plt.savefig('Figs/Fig2_A_PCs.pdf')   
    except:
        print('doing calculation for PCA plot')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Us = []
        for iN, N in enumerate(Ns):
            if iN>0:
                print('N='+str(N))
                print('--')
                for ip, output_size in enumerate(output_sizes[0:1]):
                    print(output_size)
                    
                    wout = np.zeros((N,output_size))
                    for ii in range(output_size):
                        wout[ii,ii]  = 1
            
                    input_size = 2
                    hidden_size = N
                    trials = 20
                    
                    x_init = np.random.randn(N)#x0
                    x0 = np.copy(x_init)
                    seed =21
                    sig_b = 0.
                    
                    bt = sig_b*(2*np.random.rand(N)-1.)
                    gt = 1+np.random.randn(N,1)
                    
                    dtype = torch.FloatTensor  
                    wi_init = torch.from_numpy(x_init).type(dtype)
                    wo_init = torch.from_numpy(wout).type(dtype)
                    
                    output_size2 = N
                    wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
                    
                    
                    
                    factor = 0.00001
                    Wr_ini =np.random.randn(N,N)/np.sqrt(N)
                    wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
                    b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
                    g_init_Tru = torch.from_numpy(gt).type(dtype)
                    b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
                    g_init_Ran = torch.from_numpy(factor*sigma_g*np.random.randn(N,1)+1).type(dtype)
                    #g_init_Ran = torch.from_numpy(sigma_g*np.random.randn(N,1)+1).type(dtype)
                    alpha = dta/Tau
                    input_train = np.zeros((trials, Nt, input_size))
                    input_Train = torch.from_numpy(input_train).type(dtype)
                    
                    x0s = np.random.randn(N)
                    h0_init = torch.from_numpy(x0).type(dtype)
                    
                    
                    lin_var = False
        
            
                    # NetJ         = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
                    #                                   train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                    #                                   alpha=alpha, linear=lin_var, train_h0=False)
                    # NetJ.load_state_dict(torch.load(stri+"LRnet_J_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                    
                    # NetJ_all         = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, NetJ.wrec, NetJ.b, NetJ.g, h0_init=NetJ.h0,
                    #                                                           train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                    #                                                           alpha=alpha, linear=lin_var, train_h0=False)
                                            
                    NetJ_T_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                                      train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                      alpha=alpha, linear=lin_var, train_h0=False)
        
        
                    for ilr, lr in enumerate(lrs):
                        for ic in [0,]:
                            
                            NetJ_T_all.load_state_dict(torch.load(stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                            
                            outTeacher = NetJ_T_all.forward(input_Train)
                            outteacher = outTeacher.detach().numpy()
                            
                            # outStu = NetJ_all.forward(input_Train)
                            
                            X = outteacher[10,:,:]-np.mean(outteacher[10,:,:],0)
                            C = X.T.dot(X)
                            u, v = np.linalg.eigh(C)
                            PC = v[:,-2:]
                            sol = PC.T.dot(outteacher[0,:,:].T)/np.sqrt(N)
                            plt.plot(sol[0,:], sol[1,:], color=palette2[iN])
                            
                            Us.append(u[::-1])
                            np.savez('Data/Figure3/pc_abc_N_'+str(N), u[::-1])
                        
                            
        


#%%

fig = plt.figure(figsize=[0.8*1.5*2.2*0.7, 0.8*1.5*2])
ax = fig.add_subplot(111)
for i in range(len(Ns)-1):
    plt.plot(np.arange(len(Us[i]))+1, 100*Us[i]/np.sum(Us[i]), '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,color=palette2[i+1]) #'-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, color=palette2[i])  
#plt.yscale('log')  
#plt.xscale('log')    
plt.xlim([0.5,5.5])
plt.xticks([ 1, 2, 3, 4, 5, ])
plt.yticks([0, 25, 50])
#plt.yticks([0.1, 1.])
plt.ylabel('var. expl. [%]')
plt.xlabel(r'PC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/Fig2_A_PCsvar.pdf')   

#%%
Wr = NetJ_T_all.wrec.detach().numpy()
U, S, V = np.linalg.svd(Wr)

Nsho = 8
fig = plt.figure(figsize=[0.8*1.5*2., 0.8*1.5*2])
ax = fig.add_subplot(111)
plt.imshow(Wr[0:Nsho, 0:Nsho], cmap = 'bwr', vmin =-0.02, vmax=0.02)
ax.axis('off')
plt.savefig('Figs/Fig2_connRank2.pdf') 

#%%
fig = plt.figure(figsize=[0.8*1.5*2., 0.8*1.5*2])
ax = fig.add_subplot(111)
Wr2 = np.zeros_like(Wr)
Wr2[0,1:] = U[:-1,0]*np.sqrt(S[0])
Wr2[1:,0] = V[0,:-1]*np.sqrt(S[0])
plt.imshow(Wr2[0:Nsho, 0:Nsho], cmap = 'bwr', vmin =-0.15, vmax=0.15)
ax.axis('off')
plt.savefig('Figs/Fig2_vec1Rank2.pdf')

#%%
fig = plt.figure(figsize=[0.8*1.5*2., 0.8*1.5*2])
ax = fig.add_subplot(111)
Wr2 = np.zeros_like(Wr)
Wr2[0,1:] = U[:-1,1]*np.sqrt(S[1])
Wr2[1:,0] = V[1,:-1]*np.sqrt(S[1])
plt.imshow(Wr2[0:Nsho, 0:Nsho], cmap = 'bwr', vmin =-0.15, vmax=0.15)
ax.axis('off')
plt.savefig('Figs/Fig2_vec2Rank2.pdf')


# #%%


# for iN, N in enumerate(Ns):
#     if iN==len(Ns)-4:
#         print('N='+str(N))
#         print('--')
#         for ip, output_size in enumerate(output_sizes):
#             if ip==6 or ip==0:


#                 print(output_size)
                
#                 wout = np.zeros((N,output_size))
#                 for ii in range(output_size):
#                     wout[ii,ii]  = 1
        
#                 input_size = 2
#                 hidden_size = N
#                 trials = 20
                
#                 x_init = np.random.randn(N)#x0
#                 x0 = np.copy(x_init)
#                 seed =21
#                 sig_b = 0.
                
#                 bt = sig_b*(2*np.random.rand(N)-1.)
#                 gt = 1+np.random.randn(N,1)
                
#                 dtype = torch.FloatTensor  
#                 wi_init = torch.from_numpy(x_init).type(dtype)
#                 wo_init = torch.from_numpy(wout).type(dtype)
                
#                 output_size2 = N
#                 wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
                
                
                
#                 factor = 0.00001#0.00001
#                 Wr_ini =np.random.randn(N,N)/np.sqrt(N)
#                 wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
#                 b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
#                 g_init_Tru = torch.from_numpy(gt).type(dtype)
#                 b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
#                 g_init_Ran = torch.from_numpy(factor*sigma_g*np.random.randn(N,1)+1).type(dtype)
#                 #g_init_Ran = torch.from_numpy(sigma_g*np.random.randn(N,1)+1).type(dtype)
#                 alpha = dta/Tau
#                 input_train = np.zeros((trials, Nt, input_size))
#                 input_Train = torch.from_numpy(input_train).type(dtype)
                
#                 x0s = np.random.randn(N)
#                 h0_init = torch.from_numpy(x0).type(dtype)
                
                
#                 lin_var = False
                
#                 NetJ_T_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
#                                                   train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
#                                                   alpha=alpha, linear=lin_var, train_h0=False)
#                 NetJ_T_all.load_state_dict(torch.load(stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                
#                 NetJ         = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
#                                                   train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
#                                                   alpha=alpha, linear=lin_var, train_h0=False)
#                 NetJ.load_state_dict(torch.load(stri+"LRGnet_J_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
                
#                 NetJ_all         = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, NetJ.wrec, NetJ.b, NetJ.g, h0_init=NetJ.h0,
#                                                                           train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
#                                                                           alpha=alpha, linear=lin_var, train_h0=False)
#                 NetJ_T_all.load_state_dict(torch.load(stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
#                 fig = plt.figure(figsize=[1.5*2.2, 0.8*1.5*2])
#                 ax = fig.add_subplot(111)
#                 Nex = 39
#                 for ilr, lr in enumerate(lrs):
#                     for ic in [0,]:
                        
                        
#                         outTeacher = NetJ_T_all.forward(input_Train)
#                         outteacher = outTeacher.detach().numpy()
                        
#                         outStu = NetJ_all.forward(input_Train)
#                         outstu = outStu.detach().numpy()
                        

#                         plt.plot(taus, ((outteacher[10,:,0]/np.std(outteacher[10,:,0])-outstu[10,:,0]/np.std(outstu[10,:,0]))), label='rec. neuron', lw=1.5, color=0.1*np.ones(3))
#                         plt.plot(taus, ((outteacher[10,:,Nex]/np.std(outteacher[10,:,Nex])-outstu[10,:,Nex]/np.std(outstu[10,:,Nex]))), label='unrec. neuron', lw=1.5, color=0.6*np.ones(3))

#                         plt.ylim([-0.1, 0.1])
#                         plt.xlim([0, 10])
#                         plt.xticks([0, 5, 10])
#                         plt.legend(frameon=False, loc=4)
                        

#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.yaxis.set_ticks_position('left')
#                 ax.xaxis.set_ticks_position('bottom')
#                 plt.xlabel('time')
#                 plt.ylabel(r'error activity')
#                 plt.savefig('Figs/Fig2_Dleft1_Nrec_'+str(output_size)+'.pdf')
#                 plt.show()


#                 fig = plt.figure(figsize=[1.5*2.2, 0.5*1.5*2])
#                 ax = fig.add_subplot(111)
#                 for ilr, lr in enumerate(lrs):
#                     for ic in [0,]:
                        
                        
#                         outTeacher = NetJ_T_all.forward(input_Train)
#                         outteacher = outTeacher.detach().numpy()
                        
#                         outStu = NetJ_all.forward(input_Train)
#                         outstu = outStu.detach().numpy()
                        

#                         plt.plot(taus, ((outteacher[10,:,0])), label='rec. neuron', lw=1, color=0.1*np.ones(3))
#                         plt.plot(taus, ((outteacher[10,:,Nex])), label='unrec. neuron', lw=1, color=0.6*np.ones(3))
                        
#                         #plt.plot(((outstu[10,:,0])), label='rec. neuron', lw=2, color=0.1*np.ones(3))
                        
#                         #plt.plot(((outteacher[10,:,20]-outstu[10,:,20])), label='unrec. neuron', lw=2, color=0.6*np.ones(3))

#                         #plt.ylim([-.2, .2])
#                         plt.xlim([0, 10])
#                         plt.yticks([-1.0, 0, 1.0])
#                         ax.set_yticklabels(['-1.0', '0.0', '1.0'])
#                         plt.xticks([0, 5, 10])
#                         #plt.legend(frameon=False, loc=4)
                        

#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.yaxis.set_ticks_position('left')
#                 ax.xaxis.set_ticks_position('bottom')
#                 plt.xlabel('time', color='w')
#                 plt.ylabel(r'activity')
#                 plt.savefig('Figs/Fig2_Dleft2_Nrec_'+str(output_size)+'.pdf', transparent=True)
#                 plt.show()


