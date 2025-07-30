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
#from argparse import ArgumentParser
import sys
    #%%
    
stri = 'Data/' 
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
    
    #wr = np.zeros((hidden_size, hidden_size, n_epochs))
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
            #wr[:,:,epoch] = net.cpu().wrec.detach().numpy()      
            net.to(device=device)
        else:
            bs[:,epoch]  = net.b.detach().numpy()[:,0]
            gs[:,epoch]  = net.g.detach().numpy()[:,0]
            #wr[:,:,epoch] = net.wrec.detach().numpy()
        
            
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
        return(all_losses, bs, gs,  output, output0)
    elif save_loss==False and save_params==True:
        return(bs, gs,  output, output0)
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
output_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 64,]
Ns = [100, 200, 400, 600, 800, 1000, 1200]
lrs = [0.005,]
#%%
# parser = ArgumentParser(description='Train on a task using SGD using only a few neurons as readout.')
# parser.add_argument('--seed', required=True, help='Random seed', type=int)
# parser.add_argument('--ip', required=True, help='number of neurons in readout', type=int)
# parser.add_argument('--ilr', required=True, help='learning rate', type=int)
# parser.add_argument('--iN', required=True, help='learning rate', type=int)


# args = vars(parser.parse_args())
# ic = args['seed'] #3
# ip = args['ip'] #13
# ilr = args['ilr'] #6
# iN = args['iN']
# N = Ns[iN]
# output_size = output_sizes[ip]
# lr = lrs[ilr]

# np.random.seed(21+ic)

count = 0

start_seed = 0
num_seeds = 8
ilr=0
output_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 64,]
lrs = [0.005,]#[0.001, 0.01, 0.1, 0.5, 1., 5.]
Ns = [100, 200, 400, 600, 800, 1000, 1200]
for iN_, N in enumerate(Ns):
    for seed in range(num_seeds):
        for ip_ in range(len(output_sizes)):
            if int(sys.argv[1])==(count+1):
                ic = seed
                ip = ip_
                iN = iN_
                print('seed: '+str(ic))
                print('output: '+str(output_sizes[ip]))
                print('neurons: '+str(Ns[iN]))
                
                print('count'+str(count))
            count+=1
            
            
N = Ns[iN]
output_size = output_sizes[ip]
lr = lrs[ilr]

np.random.seed(21+ic)

#%%
# =============================================================================
#   Generate network
# =============================================================================

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

#U, S, V = np.linalg.svd(Jef)
#wout = U[:,0]/np.sqrt(N)


wout = np.zeros((N,output_size))
for ii in range(output_size):
    wout[ii,ii]  = 1


#%%
dta = 0.05
taus = np.arange(0, 60, dta)
Nt = len(taus)
Tau = 0.5

read = np.zeros((len(taus), output_size))
x = np.random.randn(N)
xs = np.zeros((N, len(taus)))
xs[:,0] = x
read[0,:] = np.dot(wout.T, g[:,0]*np.tanh(x))
for it, ti in enumerate(taus[:-1]):
    x = x+dta*(-x+Jef.dot(np.tanh(x)))/Tau
    read[it+1,:] = np.dot(wout.T, g[:,0]*np.tanh(x))
    xs[:,it+1] = x

dta = 0.05
taus = np.arange(0, 20, dta)
Nt = len(taus)


#%%
x0 = xs[:,-1]
#%%
input_size = 2
hidden_size = N
trials = 60

x_init = x0

seed =21
sig_b = 0.

bt = sig_b*(2*np.random.rand(N)-1.)
gt = g

dtype = torch.FloatTensor  
wi_init = torch.from_numpy(x_init).type(dtype)
wo_init = torch.from_numpy(wout).type(dtype)

output_size2 = N
wo_init_2 = torch.from_numpy(np.diag(np.ones(N))).type(dtype)


#%%
factor = 0.00001#0.00001
Wr_ini =J
wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
g_init_Tru = torch.from_numpy(gt).type(dtype)
b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
g_init_Ran = torch.from_numpy(factor*sigma_g*np.random.randn(N,1)+1).type(dtype)
#g_init_Ran = torch.from_numpy(sigma_g*np.random.randn(N,1)+1).type(dtype)
alpha = dta/Tau
input_train = np.zeros((trials, Nt, input_size))
input_Train = torch.from_numpy(input_train).type(dtype)
#%%
x0s = np.random.randn(N)
h0_init = torch.from_numpy(x0).type(dtype)

#%%
try:
    AA = np.load(stri+'taskLR_Glosses_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
    loss = AA['arr_0']
    print(len(loss))
    if len(loss)==700:
        a=0
    else:
        print(aaa)
    print('file already exists. finish.')
    
except:
    print('doing')
    lin_var = False
    NetJ_Teacher = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                      train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                      alpha=alpha, linear=lin_var, train_h0=False)
            
    NetJ         = RNN(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
                                      train_b = False, train_g=True,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                      alpha=alpha, linear=lin_var, train_h0=False)
    
    NetJ_T_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Tru, h0_init=h0_init,
                                      train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                      alpha=alpha, linear=lin_var, train_h0=False)
    
    NetJ_0_all = RNN(input_size, hidden_size, output_size2, wi_init, wo_init_2, wrec_init_Ran, b_init_Tru, g_init_Ran, h0_init=h0_init,
                                      train_b = False, train_g=False,train_wi = False, train_wout=False, train_conn = False, noise_std=0.001,
                                      alpha=alpha, linear=lin_var, train_h0=False)
    outTeacher = NetJ_Teacher.forward(input_Train)
    outStu = NetJ.forward(input_Train)
    
    #%%
    outteacher = outTeacher.detach().numpy()
    outteacher = outTeacher
    
    outstu = outStu.detach().numpy()
    outstu = outstu
    
    plt.figure()
    plt.plot(taus, outstu[0,:,:], '--')
    plt.plot(taus, outteacher[0,:,:], c='k')
    plt.show()
    #%%
    n_ep = 7000
    mask_train = torch.from_numpy(np.ones_like(outTeacher)).type(dtype)
    #%%
    rep_n_ep = 1
    for re in range(rep_n_ep):
        print(re)
        #lossJ__, bsJ_, gsJ__, wsJ_, outJ_, outJ__ = fs.train(NetJ2, input_Train, output_TrainAll, Mask_train2, n_epochs=n_ep, plot_learning_curve=False, plot_gradient=False, 
        #                      lr=lr, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=False)
        lossJ__, bsJ, gsJ__, outJ, outJ_ = train(NetJ, input_Train, outTeacher, mask_train, n_epochs=n_ep, plot_learning_curve=False, plot_gradient=False, 
                                          lr=lr, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True)
    
        if re==0:
            lossJ_ = lossJ__
            gsJ_ = gsJ__
        else:
            lossJ_ = np.hstack((lossJ_, lossJ__))
            gsJ_ = np.hstack((gsJ_, gsJ__))
    e_gsJ_ = np.zeros_like(gsJ_)
    
    for ep in range(np.shape(gsJ_)[1]):
        e_gsJ_[:,ep] = gsJ_[:,ep]-gt[:,0]
    
    
    losses = lossJ_
    
    eG = e_gsJ_[:,::100]
    
    np.savez(stri+'taskLR_Glosses_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz', losses[::10], eG)
    try:
        np.load(stri+'taskLR_Glosses_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz')
    except:
        print('error')
        np.savez(stri+'taskLR_Glosses_percSGD_N_'+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_lr_'+str(lr)+'_unpacked.npz', losses[::10], eG)
    
    #%%
    torch.save(NetJ.state_dict(), stri+"LRGnet_J_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt')
    torch.save(NetJ_Teacher.state_dict(), stri+"LRGnet_JTeacher_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt')
    torch.save(NetJ_T_all.state_dict(), stri+"LRGnet_JTall_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt')
    torch.save(NetJ_0_all.state_dict(), stri+"LRGnet_J0all_SGD_N_"+str(N)+'_out_'+str(output_size)+'_ic_'+str(ic)+'_ip_'+str(ip)+'_lr_'+str(lr)+'.pt')



