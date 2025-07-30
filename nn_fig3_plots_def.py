#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:58:28 2022

@author: mbeiran
"""

import numpy as np

import pickle
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
        output[:,0,:] = r.matmul(self.wout)
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(self.g.T, r).matmul(self.wrec.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h+self.b.T)
            output[:,i+1,:] = r.matmul(self.wout)
        
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
# output_sizes = [1, 2, 4, 8, 16, 32, 50, 75, 100, 125, 150, 175, 200]
# lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]#[0.001, 0.01, 0.1, 0.5, 1., 5.]

# output_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 50, 75]
# lrs = [0.001, 0.02, 0.05, 0.1, 0.2]#0.5#[0.001, 0.01, 0.1, 0.5, 1., 5.]
#output_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 64, 100]
#lrs = [0.001, 0.02, 0.05, 0.1, 0.2, 0.5]#[0.001, 0.01, 0.1, 0.5, 1., 5.]
#%%
# parser = ArgumentParser(description='Train on a task using SGD using only a few neurons as readout.')
# parser.add_argument('--seed', required=True, help='Random seed', type=int)
# parser.add_argument('--ip', required=True, help='number of neurons in readout', type=int)
# parser.add_argument('--ilr', required=True, help='learning rate', type=int)

# args = vars(parser.parse_args())
# ic = args['seed'] #3
# ip = args['ip'] #13
# ilr = args['ilr'] #6

# output_size = output_sizes[ip]
# lr = lrs[ilr]

# np.random.seed(21+ic)

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

ics = np.arange(8)#np.arange(10)


output_sizes = [1, 2,  4,  8,  16, 32, 64, 128, 256, 512,]
lrs = [0.005,]#[0.001, 0.01, 0.1, 0.5, 1., 5.]
Ns = [200, 400, 600, 800, 1000]

lr = lrs[0]
#%%
try:
    sA = np.load(stri+'LRChaosnet2NoAbs_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')#'LRChaosnet2_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')
    # Problem with last N
    # losses = sA['arr_0'][:,:,:-1,:]
    # egm = sA['arr_1'][:,:,:-1,:]
    # #egm2 = sA['arr_2'][:,:,:-1,:]
    # E_Ts = sA['arr_2'][:,:,:-1,:]
    # E_knos = sA['arr_3'][:,:-1,:]
    # E_unks = sA['arr_4'][:,:-1,:]
    
    losses = sA['arr_0'][:,:,:,:]
    egm = sA['arr_1'][:,:,:,:]
    #egm2 = sA['arr_2'][:,:,:-1,:]
    E_Ts = sA['arr_2'][:,:,:,:]
    E_knos = sA['arr_3'][:,:,:]
    E_unks = sA['arr_4'][:,:,:]
    
    
    
except: 
    dtype = torch.FloatTensor
    count = 0
    for iN, N in enumerate(Ns):
        if iN>-1:
            print('N='+str(N))
            print('--')
            
            sigma_g = 0.5
            sig_b = 0
            sigma_J = 1.7
    
            output_size2 = N
            wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
    
            J = sigma_J*np.random.randn(N,N)/np.sqrt(N) 
            gt = sigma_g*np.random.randn(N,1)+1
            
    
            input_size = 2
            hidden_size = N
            trials = 20
            #Calculate x0
            dta = 0.05
            taus = np.arange(0, 80, dta)
            Nt = len(taus)
            Tau = 0.5
            
            dta = 0.05
            taus = np.arange(0, 20, dta)
            Nt = len(taus)
            
            
            x0 = np.random.randn(N)
            x_init = x0
            
            bt = sig_b*(2*np.random.rand(N)-1.)
            
            dtype = torch.FloatTensor  
            wi_init = torch.from_numpy(x_init).type(dtype)
            
            
            factor = 0.00001
            Wr_ini =J
            wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
            b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
            g_init_Tru = torch.from_numpy(gt).type(dtype)
    
    
            alpha = dta/Tau
            input_train = np.zeros((trials, Nt, input_size))
            input_Train = torch.from_numpy(input_train).type(dtype)
            
            h0_init = torch.from_numpy(x0).type(dtype)
            for ip, output_size in enumerate(output_sizes):
                print(output_size)
                
                
    
                for ic in ics:
                    error = False
                    try:
                        A = np.load(stri+'taskLR_Chaos_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
                        loss = A['arr_0'] 
                        eG = A['arr_1'] 
                        lin_var = False
                        
                
                        NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                                          train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                          alpha=alpha, linear=lin_var, train_h0=False)
                        NetJ_Teach.load_state_dict(torch.load(stri+"Chaosnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
    
                        # NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                        #                                   train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                        #                                   alpha=alpha, linear=lin_var, train_h0=False)
                        DeltaG = torch.from_numpy(eG[:,[-1,]]).type(dtype)
                        NetJ_Stu = RNN(input_size, hidden_size, output_size2, NetJ_Teach.wi,  NetJ_Teach.wout,  NetJ_Teach.wrec,  NetJ_Teach.b,  NetJ_Teach.g+DeltaG, h0_init=NetJ_Teach.h0,
                                                          train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                          alpha=alpha, linear=lin_var, train_h0=False)
    
                        
                        outTeach = NetJ_Teach.forward(input_Train)
    
                        
                        outStu = NetJ_Stu.forward(input_Train)
                        #remove this for regular error
                        outstuN= outStu.detach().numpy().reshape(-1, outStu.shape[-1])
                        outteacherN= outTeach.detach().numpy().reshape(-1, outTeach.shape[-1])
                        CC = np.corrcoef(outstuN.T, outteacherN.T)
                        CVs = np.diag(CC[0:N,N:])


                        E_kno_ = 1-np.mean((CVs[0:output_size]))
                        E_unk_ = 1-np.mean((CVs[output_size:]))
                        
                    except:
                        eG = np.nan*np.ones(N)
                        loss = np.nan
                        print('ic:'+str(ic)+' ip: '+str(output_size)+' N: '+str(N))
                        error = True
                    
                    
                    
                    
                    
    
                    if count==0:
                        losses = np.zeros((len(loss), len(output_sizes), len(Ns), len(ics)))
                        egm = np.zeros((np.shape(eG)[1], len(output_sizes), len(Ns), len(ics)))
                        #egm2 = np.zeros((2, len(output_sizes), len(Ns), len(ics)))
                        E_Ts = np.zeros((len(taus), len(output_sizes), len(Ns), len(ics)))
                        E_knos = np.zeros(( len(output_sizes), len(Ns), len(ics)))
                        E_unks = np.zeros(( len(output_sizes), len(Ns), len(ics)))
                        count = count+1
                        
                    if error :
                        #print('-- '+str(iN) + '  '+str(ic)+'  '+str(ip))
                        #print('')
                        losses[:,ip, iN, ic] = np.nan
                        egm[:,ip, iN, ic] = np.nan
                        #egm2[1,ip, iN, ic] = np.nan
                        #egm2[0,ip, iN, ic] = np.nan
                        E_Ts[:,ip,iN,ic] = np.nan
                        E_knos[ip,iN,ic] = np.nan
                        E_unks[ip,iN,ic] = np.nan
                    else:
    
                        losses[:,ip, iN, ic] = loss
                        egm[:,ip, iN, ic] = np.sqrt(np.mean(eG**2,0))
                        #gFs[:,ip, iN, ic] = gS
                        # egm2[1,ip, iN, ic] = np.sqrt(np.mean((gT-gS)**2))
                        # egm2[0,ip, iN, ic] = np.sqrt(np.mean((gT-g0)**2))
                        #E_Ns[:,ip,iN,ic] = E_N
                        E_Ts[:,ip,iN,ic] = ((outTeach-outStu)**2).mean().detach().numpy()
                        E_knos[ip,iN,ic] = E_kno_#((outTeach[:,:,0:output_size]-outStu[:,:,0:output_size])**2).mean().detach().numpy()

                        E_unks[ip,iN,ic] = E_unk_#((outTeach[:,:,output_size:]-outStu[:,:,output_size:])**2).mean().detach().numpy()
                        print(E_unk_)
    
    

    np.savez('LRChaosnet2NoAbs_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz', losses, egm, E_Ts, E_knos, E_unks)
#%%


try:
    sA = np.load(stri+'LRChaosnet2ZeroNoAbs_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')#'LRChaosnet2_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')
    # Problem with last N
    E0s = sA['arr_0']

    
except: 
    print('calculating point zero')
    E0s = np.zeros((  len(Ns), len(ics)))
    dtype = torch.FloatTensor
    count = 0
    for iN, N in enumerate(Ns):
        if iN>-1:
            print('N='+str(N))
            print('--')
            
            sigma_g = 0.5
            sig_b = 0
            sigma_J = 1.7
    
            output_size2 = N
            wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
    
            J = sigma_J*np.random.randn(N,N)/np.sqrt(N) 
            gt = sigma_g*np.random.randn(N,1)+1
            
    
            input_size = 2
            hidden_size = N
            trials = 20
            #Calculate x0
            dta = 0.05
            taus = np.arange(0, 80, dta)
            Nt = len(taus)
            Tau = 0.5
            
            dta = 0.05
            taus = np.arange(0, 20, dta)
            Nt = len(taus)
            
            
            x0 = np.random.randn(N)
            x_init = x0
            
            bt = sig_b*(2*np.random.rand(N)-1.)
            
            dtype = torch.FloatTensor  
            wi_init = torch.from_numpy(x_init).type(dtype)
            
            
            factor = 0.00001
            Wr_ini =J
            wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
            b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
            g_init_Tru = torch.from_numpy(gt).type(dtype)
    
    
            alpha = dta/Tau
            input_train = np.zeros((trials, Nt, input_size))
            input_Train = torch.from_numpy(input_train).type(dtype)
            
            h0_init = torch.from_numpy(x0).type(dtype)
            ip = 0
            output_size = output_sizes[ip]
            

            for ic in ics:
                error = False
                A = np.load(stri+'taskLR_Chaos_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
                loss = A['arr_0'] 
                eG = A['arr_1'] 
                lin_var = False
                
        
                NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                                  train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                  alpha=alpha, linear=lin_var, train_h0=False)
                NetJ_Teach.load_state_dict(torch.load(stri+"Chaosnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))

                # NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                #                                   train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                #                                   alpha=alpha, linear=lin_var, train_h0=False)
                DeltaG = torch.from_numpy(eG[:,[0,]]).type(dtype)
                NetJ_Stu = RNN(input_size, hidden_size, output_size2, NetJ_Teach.wi,  NetJ_Teach.wout,  NetJ_Teach.wrec,  NetJ_Teach.b,  NetJ_Teach.g+DeltaG, h0_init=NetJ_Teach.h0,
                                                  train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                  alpha=alpha, linear=lin_var, train_h0=False)

                
                outTeach = NetJ_Teach.forward(input_Train)

                
                outStu = NetJ_Stu.forward(input_Train)
                #remove this for regular error
                outstuN= outStu.detach().numpy().reshape(-1, outStu.shape[-1])
                outteacherN= outTeach.detach().numpy().reshape(-1, outTeach.shape[-1])
                CC = np.corrcoef(outstuN.T, outteacherN.T)
                CVs = np.diag(CC[0:N,N:])


                E_ = 1-np.mean((CVs))#np.mean(np.abs(CVs))

                    
                E0s[iN,ic] = E_#((outTeach[:,:,0:output_size]-outStu[:,:,0:output_size])**2).mean().detach().numpy()


    

    np.savez('LRChaosnet2ZeroNoAbs_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz', E0s)

#%%

#Calculate average error
try:
    sA = np.load(stri+'avg_ErefsNoAbs.npz')#'LRChaosnet2_sevN_Ns_'+str(len(Ns))+'_out_'+str(len(output_sizes))+'_ic_'+str(len(ics))+'.npz')
    # Problem with last N
    E_refs = sA['arr_0']
    E_Grefs = sA['arr_1']
except: 
    perms = 10
    E_refs  = np.zeros((len(Ns), len(ics), perms))
    E_Grefs  = np.zeros((len(Ns), len(ics), perms))
    
    for iN, N in enumerate(Ns):
        if iN>-1:
            print('N='+str(N))
            print('--')
            
            sigma_g = 0.5
            sig_b = 0
            sigma_J = 1.7
    
            output_size2 = N
            wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
    
            J = sigma_J*np.random.randn(N,N)/np.sqrt(N) 
            gt = sigma_g*np.random.randn(N,1)+1
            
    
            input_size = 2
            hidden_size = N
            trials = 20
            #Calculate x0
            dta = 0.05
            taus = np.arange(0, 80, dta)
            Nt = len(taus)
            Tau = 0.5
            
            dta = 0.05
            taus = np.arange(0, 20, dta)
            Nt = len(taus)
            
            
            x0 = np.random.randn(N)
            x_init = x0
            
            bt = sig_b*(2*np.random.rand(N)-1.)
            
            dtype = torch.FloatTensor  
            wi_init = torch.from_numpy(x_init).type(dtype)
            
            
            factor = 0.00001
            Wr_ini =J
            wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
            b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
            g_init_Tru = torch.from_numpy(gt).type(dtype)
    
    
            alpha = dta/Tau
            input_train = np.zeros((trials, Nt, input_size))
            input_Train = torch.from_numpy(input_train).type(dtype)
            
            h0_init = torch.from_numpy(x0).type(dtype)
            for ip, output_size in enumerate(output_sizes[0:1]):
                print(output_size)
                for ic in ics:
                    error = False
                    try:
    
                        lin_var = False
                        
                
                        NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                                          train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                                          alpha=alpha, linear=lin_var, train_h0=False)
                        NetJ_Teach.load_state_dict(torch.load(stri+"Chaosnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
    
                       
                        outTeach = NetJ_Teach.forward(input_Train)
    
                        for ipe in range(perms):
                            #outStu = NetJ_Stu.forward(input_Train)
                            outStu = outTeach[:,:,np.random.permutation(N)]
                            
                            #remove this for regular error
                            outstuN= outStu.detach().numpy().reshape(-1, outStu.shape[-1])
                            outteacherN= outTeach.detach().numpy().reshape(-1, outTeach.shape[-1])
                            CC = np.corrcoef(outstuN.T, outteacherN.T)
                            CVs = np.diag(CC[0:N,N:])
                            E_refs[iN, ic, ipe] =  1-np.mean((CVs))
                            E_Grefs[iN, ic, ipe] =  np.sqrt(np.mean((gt-gt[np.random.permutation(N)])**2,0))
                            
                    except:
                        eG = np.nan*np.ones(N)
                        loss = np.nan
                        print('ic:'+str(ic)+' ip: '+str(output_size)+' N: '+str(N))
                        error = True
                        E_refs[iN, ic, :] =  np.nan
                    
    np.savez('avg_ErefsNoAbs.npz', E_refs, E_Grefs)

#%%
palette2 = plt.cm.copper(np.linspace(0, 0.8, len(Ns)))
    
fig = plt.figure()
ax = fig.add_subplot(111)
# for IC in ics:
#     plt.plot(output_sizes, E_knos[:,iN,IC]/E_unks[:,iN,IC], c=0.5*np.ones(3))
    

for iN in np.arange(len(Ns)):
    Y = E_knos[:,iN,:]/E_unks[:,iN,:]
    MY = np.mean(Y,-1)
    SY = np.std(Y,-1)/np.sqrt(len(ics))
    plt.plot(output_sizes, MY, c=palette2[iN], lw=2., label='N='+str(Ns[iN]))    
    plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)
plt.legend()
plt.xscale('log')
plt.ylabel('error known/ unknown')
plt.xlabel('# known neurons')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('SGD_Chaos_sevN_errorRatio_vs_M.pdf')

#%%
palette2 = plt.cm.copper(np.linspace(0, 0.8, len(Ns)))

LW = 1
IC = 0
iN = 0
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)

for iN in np.arange(len(Ns)):

    Y = np.copy(E_knos[:,iN,:])#/E_knos[:,iN,0][:,np.newaxis]
    #print(np.sum(Y>0.001))
    #Y[Y>0.001] = np.nan
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

plt.ylim([0.0005, 1.9])
#plt.yticks([0.01,0.03])
#plt.yscale('log')

plt.ylabel(r'$1-\left[\rho\right]$ (recorded act.)')
plt.xlabel('rec. units $M$')
#plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
plt.savefig('Figs/F2_ChaosA_recorded.pdf', transparent=True)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)


for iN in np.arange(len(Ns)):
    Y = egm[-1,:,iN,:]#/egm[0,iPP,iN,:]
    MY = np.mean(Y,-1)
    SY = np.std(Y,-1)/np.sqrt(len(ics))
    XX = np.arange(len(MY))
    plt.plot(output_sizes, MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, c=palette2[iN], lw=3., label='N='+str(Ns[iN]))    
    plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.2)
    Y0 = egm[0,0,iN,:]#/egm[0,iPP,iN,:]
    MY0 = np.nanmean(Y0)
    SY0 = np.nanmean(Y0)
    #plt.fill_between([0.5,], MY0-SY0, MY0+SY0, color=palette2[iN], alpha=0.2)
    plt.scatter(0.5, MY0, edgecolor='k', linewidth=0.5, c=palette2[iN],s=20)

plt.xscale('log')
plt.xticks([0.5, 1, 10, 100, 1000])
ax.set_xticklabels(['0', 1, 10, 100, 1000])

xl = ax.get_xlim()
plt.plot(xl, np.ones_like(xl)*np.mean(E_Grefs), lw=3, alpha=0.3, c='k')
plt.yscale('log')
plt.yticks([0.01, 0.1, 1.])
plt.ylim([0.008, 1.1])
plt.ylabel(r'error gains')
plt.xlabel('rec. units $M$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/F2_ChaosA_gains.pdf')

#%%
Wr = np.random.randn(N,N)/np.sqrt(N)
U, S, V = np.linalg.svd(Wr)

Nsho = 8
fig = plt.figure(figsize=[0.8*1.5*2., 0.8*1.5*2])
ax = fig.add_subplot(111)
plt.imshow(Wr[0:Nsho, 0:Nsho], cmap = 'bwr', vmin =-0.09, vmax=0.09)
ax.axis('off')
plt.savefig('Figs/F2_connRankFullR.pdf') 

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
grey = 0.5*np.ones(3)


for iN in np.arange(len(Ns)):
    if iN>-1:
        Ym = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        Y = E_unks[:,iN,:]#/E_unks[:,iN,0][:,np.newaxis]
        
        # try:
        MY = np.nanmean(Y,-1)
        MY[MY<1e-4]=np.nan
        # print(Ns[iN])
        # print(MY)
        # print('--')
        SY = np.nanstd(Y,-1)/np.sqrt(np.sum(~np.isnan(Y),-1))
        plt.plot(output_sizes, MY, '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,  c=palette2[iN], lw=2.,  label=str(Ns[iN]))    
        plt.fill_between(output_sizes, MY-SY, MY+SY, color=palette2[iN], alpha=0.5)
        
        MY = np.nanmean(E0s,-1)
        MY[MY<1e-4]=np.nan
        # print(Ns[iN])
        # print(MY)
        # print('--')
        SY = np.nanstd(E0s,-1)/np.sqrt(np.sum(~np.isnan(E0s),-1))
        
        # MY2 = np.nanmean(E0s[iN,:])
        # SY2 = np.nanstd(E0s[iN,:])/np.sqrt(len(ics))
        # print(MY2)
        # plt.scatter(0.5, MY2, c=palette2[iN], s=25, edgecolor='k', linewidth=0.5,)
        # plt.plot([0.5, 0.5], [MY2-SY2, MY2+SY2])
        
        
        # except:
        #     print('not possible')
xl = ax.get_xlim()
#plt.plot(xl, np.ones(2)*np.mean(E_refs), lw=3, alpha=0.3, c='k')        
plt.plot(xl, 0.2*np.ones_like(xl), '--', c='k')      
plt.legend(frameon=False, fontsize=10)
plt.xscale('log')
plt.yscale('log')
xtickslocs = ax.get_xticks()

for iN in np.arange(len(Ns)):
    if iN>-1:
        
        MY2 = np.nanmean(E0s[iN,:])
        SY2 = np.nanstd(E0s[iN,:])/np.sqrt(len(ics))
        print(MY2)
        plt.scatter(0.4, MY2, c=palette2[iN], s=25, edgecolor='k', linewidth=0.5,)
        plt.plot([0.4, 0.4], [MY2-SY2, MY2+SY2])


#ax.set_xticks(xtickslocs)
# #plt.ylim([0.00001, 0.01])
# xl = ax.get_xlim()
# #plt.plot(xl, np.ones_like(xl)*np.mean(E_refs), lw=3, alpha=0.3, c='k')
# ax.set_xlim(xl)
ax.set_ylim([0.0005, 1.9])

xticks = np.hstack((np.arange(0.5, 1., 0.1), np.arange(1., 10, 1.), np.arange(10, 100, 10), np.arange(100, 1000, 100), np.arange(1000, 2000, 1000)))

ax.set_xticks(xticks)
plt.ylabel(r'$1-\left[\rho\right]$ (unrecorded act.)')
plt.xlabel('rec. units $M$')
#plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(percs[::2])
plt.savefig('Figs/F2_ChaosA_unrecorded.pdf', transparent=True)
plt.show()
#%%

Mast = np.zeros(len(Ns))
for iN in np.arange(len(Ns)):
    if iN>-1:
        Ym = E_knos[:,iN,:]#/E_knos[:,iN,0][:,np.newaxis]
        Y = E_unks[:,iN,:]#/E_unks[:,iN,0][:,np.newaxis]
        
        # try:
        
        MY = np.nanmean(Y,-1)

        Xs = output_sizes
        Xs2 = np.arange(1, 1000)
        MY[np.isnan(MY)]=0.
        MY2 = np.interp(Xs2, Xs, MY)

        II = np.argmin(np.abs(MY2-0.2))
        Mast[iN] = Xs2[II]

        plt.plot(Xs2, MY2, color=palette2[iN])
        plt.scatter(Xs2[II], MY2[II], s=40)
        plt.yscale('log')
        plt.xscale('log')
#%%
fig = plt.figure(figsize=[0.8*1.5*2.2*0.7, 0.8*1.5*2*0.7])
ax = fig.add_subplot(111)
plt.plot(Ns, Mast,  c='k')
for iN in np.arange(len(Ns)):
    plt.scatter(Ns[iN], Mast[iN], edgecolor='k', color=palette2[iN], s=30, zorder=4)

ax.set_xlim([10, 1100])
ax.set_ylim([1, 140])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'network size $N$')
ax.set_ylabel(r'$M^\ast$')
plt.savefig('Figs/F2_ChaosA_unrecordedSumm.pdf', transparent=True)
#plt.plot(Ns, 0.15*np.array(Ns), c='k')



#%%

ic = 6 # I like 4
ip = 1
try:
    bb = np.load(stri+'chaos_samples.npz')
    #Uxs = bb['arr_0']
    Prs = bb['arr_0']
    Acts = bb['arr_1']
    dtype = torch.FloatTensor  
    Uxs = []
    for iN, N in enumerate(Ns):
        if iN>-1:
            Ux = np.load(stri+'Ux_N_'+str(N)+'.npz')
            Uxs.append(Ux['arr_0'])
            
except:
    print('calculating chaotic examples')
    
    Uxs = []
    Prs = []
    
    dtype = torch.FloatTensor  
    for iN, N in enumerate(Ns):
        if iN>-1:
            print('N='+str(N))
            print('--')
            
            sigma_g = 0.5
            sig_b = 0
            sigma_J = 1.7
    
            output_size2 = N
            wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
    
            J = sigma_J*np.random.randn(N,N)/np.sqrt(N) 
            gt = sigma_g*np.random.randn(N,1)+1
            
    
            input_size = 2
            hidden_size = N
            trials = 20
            #Calculate x0
            dta = 0.05
            taus = np.arange(0, 80, dta)
            Nt = len(taus)
            Tau = 0.5
            
            dta = 0.05
            taus = np.arange(0, 20, dta)
            Nt = len(taus)
            
            
            x0 = np.random.randn(N)
            x_init = x0
            
            bt = sig_b*(2*np.random.rand(N)-1.)
            
            dtype = torch.FloatTensor  
            wi_init = torch.from_numpy(x_init).type(dtype)
            
            
            factor = 0.00001
            Wr_ini =J
            wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
            b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
            g_init_Tru = torch.from_numpy(gt).type(dtype)
    
    
            alpha = dta/Tau
            input_train = np.zeros((trials, Nt, input_size))
            input_Train = torch.from_numpy(input_train).type(dtype)
            
            h0_init = torch.from_numpy(x0).type(dtype)
            
            
            A = np.load(stri+'taskLR_Chaos_percSGD_N_'+str(N)+'_out_'+str(output_sizes[ip])+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
            loss = A['arr_0'] 
            eG = A['arr_1'] 
            lin_var = False
            
    
            NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                              train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                              alpha=alpha, linear=lin_var, train_h0=False)
            NetJ_Teach.load_state_dict(torch.load(stri+"Chaosnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_sizes[ip])+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
    
            # NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
            #                                   train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
            #                                   alpha=alpha, linear=lin_var, train_h0=False)
            DeltaG = torch.from_numpy(eG[:,[-1,]]).type(dtype)
            NetJ_Stu = RNN(input_size, hidden_size, output_size2, NetJ_Teach.wi,  NetJ_Teach.wout,  NetJ_Teach.wrec,  NetJ_Teach.b,  NetJ_Teach.g+DeltaG, h0_init=NetJ_Teach.h0,
                                              train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                              alpha=alpha, linear=lin_var, train_h0=False)
    
            
            outTeach = NetJ_Teach.forward(input_Train)
            outteacherN= outTeach.detach().numpy().reshape(-1, outTeach.shape[-1])
            if iN==0:
                Acts = np.zeros((len(Ns), np.shape(outTeach)[1],10))
                
            Acts[iN,:,:] = (outTeach[0,:,0:10].detach().numpy().astype('float64'))
            Xx = outteacherN-np.mean(outteacherN,axis=0, keepdims=True)
            C = Xx.T.dot(Xx)
            Ux, Vx = np.linalg.eigh(C)
            Ux = Ux[::-1]
            Vx = Vx[::-1]
            Uxs.append(Ux)
            Prs.append(np.sum(Ux)**2/np.sum(Ux**2))
            np.savez(stri+'Ux_N_'+str(N), Ux)
    np.savez(stri+'chaos_samples.npz', Prs, Acts)
    
# #%%
# IPs = [2, 6, 8]
# CLS = [[42/255, 6/255, 147/255], [161/255, 27/255, 155/255], [248/255, 149/255, 64/255]]
# for iN, N in enumerate(Ns):
#     if N==Ns[1] or N==Ns[-1]:
#         output_size2 = N
        
                
#         sigma_g = 0.5
#         sig_b = 0
#         sigma_J = 1.7

#         output_size2 = N
#         wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)

#         J = sigma_J*np.random.randn(N,N)/np.sqrt(N) 
#         gt = sigma_g*np.random.randn(N,1)+1
        

#         input_size = 2
#         hidden_size = N
#         trials = 20
#         #Calculate x0
#         dta = 0.05
#         taus = np.arange(0, 80, dta)
#         Nt = len(taus)
#         Tau = 0.5
        
#         dta = 0.05
#         taus = np.arange(0, 20, dta)
#         Nt = len(taus)
        
        
#         x0 = np.random.randn(N)
#         x_init = x0
        
#         bt = sig_b*(2*np.random.rand(N)-1.)
        
#         dtype = torch.FloatTensor  
#         wi_init = torch.from_numpy(x_init).type(dtype)
        
        
#         factor = 0.00001
#         Wr_ini =J
#         wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
#         b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
#         g_init_Tru = torch.from_numpy(gt).type(dtype)


#         alpha = dta/Tau
#         input_train = np.zeros((trials, Nt, input_size))
#         input_Train = torch.from_numpy(input_train).type(dtype)
        
#         h0_init = torch.from_numpy(x0).type(dtype)
#         wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)
        
#         NetJ_Teach = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
#                                           train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
#                                           alpha=alpha, linear=lin_var, train_h0=False)
#         NetJ_Teach.load_state_dict(torch.load(stri+"Chaosnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_sizes[ip])+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt', map_location='cpu'))
        
#         outTeach = NetJ_Teach.forward(input_Train)

#         fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
#         ax = fig.add_subplot(111)
#         for ip0, ip in enumerate(IPs):
#             print(output_sizes[ip])
#             A = np.load(stri+'taskLR_Chaos_percSGD_N_'+str(N)+'_out_'+str(output_sizes[ip])+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
#             loss = A['arr_0'] 
#             eG = A['arr_1'] 
#             lin_var = False
            
#             DeltaG = torch.from_numpy(eG[:,[-1,]]).type(dtype)
#             NetJ_Stu = RNN(input_size, hidden_size, output_size2, NetJ_Teach.wi,  NetJ_Teach.wout,  NetJ_Teach.wrec,  NetJ_Teach.b,  NetJ_Teach.g+DeltaG, h0_init=NetJ_Teach.h0,
#                                               train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
#                                               alpha=alpha, linear=lin_var, train_h0=False)
            
#             outStu = NetJ_Stu.forward(input_Train)
#             outstuN= outStu.detach().numpy()
#             outteaN= outTeach.detach().numpy()
            
#             tt = np.arange(len(outstuN[0,:,-2]))*0.1
#             plt.plot(tt, outstuN[0,:,-2].T, lw=2, label='M='+str(output_sizes[ip]), color=CLS[ip0])
#         plt.plot(tt, outteaN[0,:,-2].T, lw=5, alpha=0.3, c='k', label='teacher')                
#         plt.legend(loc=4)    
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         ax.set_xlim([0, 49])
#         ax.set_ylim([-2.4, 1.5])
        
#         ax.set_xlabel(r'time ($\tau$)')
#         ax.set_ylabel(r'activity unrecorded neuron')
#         plt.savefig('Figs/F2_ChaosA_PC_sample2_N_'+str(N)+'.pdf')  
        
#         plt.show()
        
#%%
fig = plt.figure(figsize=[0.8*1.5*2.2, 0.8*1.5*2])
ax = fig.add_subplot(111)
for i in range(len(Ns)):
    plt.plot(taus, Acts[i][:,0:4]-3*i, lw=2, color=palette2[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticklabels('')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time (a.u.)')
plt.savefig('Figs/F2_ChaosA_PC_sample.pdf')  
#%%
fig = plt.figure(figsize=[0.8*1.5*2.2*0.7, 0.8*1.5*2])
ax = fig.add_subplot(111)


for i in range(len(Ns)):
    plt.plot(np.arange(len(Uxs[i]))+1, 100*Uxs[i]/np.sum(Uxs[i]), '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,color=palette2[i], label=str(Ns[i])) #'-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, color=palette2[i])  
#plt.legend(frameon=False, fontsize=10)
#plt.yscale('log')  
#plt.xscale('log')    
plt.xlim([0.5,5.5])
plt.xticks([ 1,  2, 3, 4, 5, ])
plt.yticks([0, 25, 50])
#plt.yticks([0.1, 1.])
plt.ylabel('var. expl. [%]')
plt.xlabel(r'PC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/F2_ChaosA_PCsvar.pdf')  

#%%
fig = plt.figure(figsize=[0.8*1.5*2.2, 0.8*1.5*2])
ax = fig.add_subplot(111)


for i in range(len(Ns)):
    plt.plot(np.arange(len(Uxs[i]))+1, 100*Uxs[i]/np.sum(Uxs[i]), '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,color=palette2[i], label=str(Ns[i])) #'-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, color=palette2[i])  
plt.legend(frameon=False, fontsize=10)
#plt.yscale('log')  
#plt.xscale('log')    
plt.xlim([0.5,10.5])
plt.xticks([ 1,  3,  5, 7, 9])
plt.yticks([0, 25, 50])
#plt.yticks([0.1, 1.])
plt.ylabel('var. expl. [%]')
plt.xlabel(r'PC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/F2_ChaosA_PCs_PR.pdf')  


#%%
fig = plt.figure(figsize=[0.8*1.5*2.2, 0.8*1.5*2])
ax = fig.add_subplot(111)
for i in range(len(Ns)):
    plt.plot(np.arange(len(Uxs[i]))+1, 100*Uxs[i]/np.sum(Uxs[i]), '-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5,color=palette2[i], label=str(Ns[i])) #'-o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, color=palette2[i])  
plt.legend(frameon=False, fontsize=10)
#plt.yscale('log')  
#plt.xscale('log')    
plt.xlim([0.5,10.5])
plt.xticks([ 1,  3,  5, 7, 9])
plt.yticks([0, 25, 50])
#plt.yticks([0.1, 1.])
plt.ylabel('var. expl. [%]')
plt.xlabel(r'PC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Figs/F2_ChaosA_PCsvar_.pdf')  



