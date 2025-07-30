#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:24:12 2023

@author: mbeiran
"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time
import random
from math import sqrt

def set_plot():
    plt.style.use('ggplot')
    
    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.dpi'] = 300
     
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
    plt.rcParams["axes.grid"] = False
    return()

def create_input(taus, trials, omega = 3):
    Nt = len(taus)
    input_train = np.zeros((trials, Nt, 2))
    output_train = np.zeros((trials, Nt, 1))
    cond_train = np.zeros((trials))

    To = 2*np.pi/omega
    for tr in range(trials):
        tI = 2+3*np.random.rand()
        iI = np.argmin(np.abs(taus-tI))
        cond = np.random.randint(2)
        input_train[tr,iI,cond] = 1. 
        
        output_train[tr,iI:,0] = np.sin(omega*(taus[iI:]-taus[iI])) 
        if cond==0:
            iF = np.argmin(np.abs(taus-tI-To*2))
            output_train[tr,iF:,0] = 0.
        else:
            iF = np.argmin(np.abs(taus-tI-To*5))
            output_train[tr,iF:,0] = 0.
        if np.random.rand()<0.2:
            input_train[tr,:,:] = 0.
            output_train[tr,:,0] = 0
            cond = 3
        cond_train[tr] = cond
    
    return(input_train, output_train, cond_train)

#%%
# RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, mask_wrec, refEI, b_init, g_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_h0 = True, train_wout = True, noise_std=0., alpha=0.2, linear=False, beta=1.):
        
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_b = train_b
        self.train_g = train_g
        self.train_conn = train_conn
        self.beta = beta
        if linear==True:    
            self.non_linearity = torch.clone
        else:
            self.non_linearity = torch.nn.functional.softplus
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.wi.requires_grad= False
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
            
        self.mwrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.mwrec.requires_grad= False
        
        self.refEI = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.refEI.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if train_h0:
            self.h0.requires_grad = True
        else:
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
            self.mwrec.copy_(mask_wrec)
            self.refEI.copy_(refEI)
            if h0_init is None:
                self.h0.zero_
                self.h0.fill_(1)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0+self.b.T, beta=self.beta)
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = r.matmul(self.wout)
        # clip_weights
        if torch.any(self.refEI==1.):
            effW_ = torch.relu(torch.mul(self.wrec, self.refEI))
            effW_ = torch.mul(effW_, self.refEI)
            effW_ = effW_+torch.mul(-1*self.wrec, torch.abs(self.refEI)-1)
        else:
            effW_ = self.wrec
        effW = torch.mul(effW_, self.mwrec)
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(torch.relu(self.g.T), r).matmul(effW.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h+self.b.T, beta=self.beta)
            output[:,i+1,:] = r.matmul(self.wout)
        return output

class RNNgain(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, mask_wrec, refEI, b_init, g_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_h0 = True, train_wout = True, noise_std=0., alpha=0.2, linear=False, beta=1.):
        
        super(RNNgain, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_b = train_b
        self.train_g = train_g
        self.train_conn = train_conn
        self.beta = beta
        if linear==True:    
            self.non_linearity = torch.clone
        else:
            self.non_linearity = torch.nn.functional.softplus
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.wi.requires_grad= False
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
            
        self.mwrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.mwrec.requires_grad= False
        
        self.refEI = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.refEI.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if train_h0:
            self.h0.requires_grad = True
        else:
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
            self.mwrec.copy_(mask_wrec)
            self.refEI.copy_(refEI)
            if h0_init is None:
                self.h0.zero_
                self.h0.fill_(1)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0+self.b.T, beta=self.beta)
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = torch.mul(torch.relu(self.g.T), r).matmul(self.wout)
        # clip_weights
        if torch.any(self.refEI==1.):
            effW_ = torch.relu(torch.mul(self.wrec, self.refEI))
            effW_ = torch.mul(effW_, self.refEI)
            effW_ = effW_+torch.mul(-1*self.wrec, torch.abs(self.refEI)-1)
        else:
            effW_ = self.wrec
        effW = torch.mul(effW_, self.mwrec)
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(torch.relu(self.g.T), r).matmul(effW.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h+self.b.T, beta=self.beta)
            output[:,i+1,:] = torch.mul(torch.relu(self.g.T), r).matmul(self.wout)
        return output


class RNNgainBeta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, mask_wrec, refEI, b_init, g_init, beta_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_h0 = True, train_wout = True, train_beta=True, noise_std=0., alpha=0.2):
        
        super(RNNgainBeta, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_b = train_b
        self.train_g = train_g
        self.train_conn = train_conn
        #self.beta = beta
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.wi.requires_grad= False
        
        self.beta = nn.Parameter(torch.Tensor(1))
        if train_beta:
            self.beta.requires_grad = True
        else:
            self.beta.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
            
        self.mwrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.mwrec.requires_grad= False
        
        self.refEI = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.refEI.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if train_h0:
            self.h0.requires_grad = True
        else:
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
            self.mwrec.copy_(mask_wrec)
            self.refEI.copy_(refEI)
            self.beta.copy_(beta_init)
            if h0_init is None:
                self.h0.zero_
                self.h0.fill_(1)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.h0+self.b.T
        Thres = 30.
        mask = self.beta*r<Thres
        r[mask] = torch.pow(self.beta, -1).mul(torch.log(1+torch.exp(self.beta.mul(r[mask])))) 
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = torch.mul(torch.relu(self.g.T), r).matmul(self.wout)
        # clip_weights
        if torch.any(self.refEI==1.):
            effW_ = torch.relu(torch.mul(self.wrec, self.refEI))
            effW_ = torch.mul(effW_, self.refEI)
            effW_ = effW_+torch.mul(-1*self.wrec, torch.abs(self.refEI)-1)
        else:
            effW_ = self.wrec
        effW = torch.mul(effW_, self.mwrec)
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(torch.relu(self.g.T), r).matmul(effW.t())+ input[:,i,:].matmul(self.wi))
        
            r = h+self.b.T
            mask = r<Thres
            r[mask] = torch.pow(self.beta, -1).mul(torch.log(1+torch.exp(self.beta.mul(r[mask])))) 
            
            #r = torch.pow(self.beta, -1).mul(torch.log(1+torch.exp(self.beta.mul(h+self.b.T)))) #self.non_linearity(h+self.b.T, beta=self.beta)
            output[:,i+1,:] = torch.mul(torch.relu(self.g.T), r).matmul(self.wout)
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
          clip_gradient=None, keep_best=False, cuda=False, save_loss=False, save_params=True, verbose=True, adam=True, train_weight=False, skip_ep=10, train_beta=False):
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
    hidden_size = net.hidden_size
    if not train_weight:
        bs = np.zeros((hidden_size, n_epochs))
        gs = np.zeros((hidden_size, n_epochs))
    # graD = np.zeros((hidden_size, n_epochs))
    else:
        wr = np.zeros((hidden_size, hidden_size, n_epochs//skip_ep))

    if train_beta:
        betas = np.zeros((n_epochs))
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
        # check if there is any nan in output
        if torch.isnan(output).any():
            print("Nan in output")
            break
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
            if not train_weight:
                bs[:,epoch]  = net.cpu().b.detach().numpy()[:,0]
                gs[:,epoch]  = net.cpu().g.detach().numpy()[:,0]
            elif np.mod(epoch,skip_ep)==0:
                    wr[:,:,epoch//skip_ep] = net.cpu().wrec.detach().numpy()      
            if train_beta:
                betas[epoch] = net.cpu().beta.detach().numpy()
            net.to(device=device)
        else:
            if not train_weight:
                bs[:,epoch]  = net.b.detach().numpy()[:,0]
                gs[:,epoch]  = net.g.detach().numpy()[:,0]
            elif np.mod(epoch,skip_ep)==0:
                wr[:,:,epoch//skip_ep] = net.wrec.detach().numpy()
            if train_beta:
                betas[epoch] = net.beta.detach().numpy()
            
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
        if not train_weight:
            if not train_beta:
                return(all_losses, bs, gs, output, output0)
            else:
                return(all_losses, bs, gs, betas, output, output0)
        else:
            return(all_losses, wr, output, output0)
    elif save_loss==False and save_params==True:
        
        if not train_weight:
            return(bs, gs, output, output0)
        else:
            return(wr, output, output0)
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


def softpl(x, beta=1):
    return ((1/beta)*np.log(1.+np.exp(beta*x)))

def relu(x):
    return (x*(x>0.))

def d_soft(x):
    return (np.exp(x)/(1+np.exp(x)))

def get_losses(input_train, output_train, taus, net, gs, bs, Nkno, every_ep=10, every_t = 4, calculate=False, cpu=False, relu_=False, sort=False, gain=False, beta=1., cc=False, readout=False, readout_wout=0):
    if not cpu:
        Net = net
    else:
        Net = net.cpu()
    x0 = Net.h0.detach().numpy()
    J = Net.wrec.detach().numpy()
    refEI = Net.refEI.detach().numpy()
    mwrec = Net.mwrec.detach().numpy()
    wi = Net.wi.detach().numpy()
    if len(beta)>1:
        train_beta = True
    else:
        train_beta = False

    N = Net.hidden_size
    loss_unk = []  
    loss_kno = []
    if cc:
        loss_unk_R = []  
        loss_kno_R = []        
    dta = taus[1]-taus[0]
    for epo in range(np.shape(gs)[-1]):
        if np.mod(epo, every_ep)==0:
            x = x0
            tri = np.shape(output_train)[0]
            x = np.zeros((tri, np.shape(output_train)[-1]))
            if np.mod(epo, 20)==0:
                print(epo)
            
            out_ = np.zeros((tri, N))
            loss_kno_=0
            loss_unk_=0

            x[0:N,:] = x0
            if relu_:
                r=relu(x+bs[:,epo])
            elif train_beta:
                r = 1./beta[epo]*np.log(1+np.exp(beta[epo]*(x+bs[:,epo])))
            else:
                r = softpl(x+bs[:,epo], beta=beta)
            c_t = 0
            if calculate or cc:
                out_s = np.zeros_like(output_train)
                
            if refEI[0,0]!=0:
                effW_ = relu(J*refEI)
                effW_ = effW_ * refEI
            else:
                effW_ = np.copy(J)
            effW = effW_ * mwrec
            

            for it, ta in enumerate(taus):
                #out_[:,0] = np.dot(wout, r.T)
                if np.sum(sort)==0:
                    if not gain:
                        out_= r
                    else:
                        out_ = gs[:,epo]*r
                        # if epo==0:
                        #     print(np.shape(gs))
                        #     print(np.shape(gs[:,epo]))
                else:
                    if not gain:
                        out_ = r[:,sort]
                    else:
                        out_ = gs[sort,epo]*r[:,sort]
                    
                if calculate or cc:
                    out_s[:,it,:] = out_
                    
                #torch.mul(torch.relu(self.g.T), r).matmul(effW.t())
                x=x+dta*(-x+(r.dot(np.diag(relu(gs[:,epo])))).dot(effW.T)+input_train[:,it,:].dot(wi))
                if relu_:
                    r=relu(x+bs[:,epo])
                elif train_beta:
                    r = 1./beta[epo]*np.log(1+np.exp(beta[epo]*(x+bs[:,epo])))
                else:
                    r = softpl(x+bs[:,epo], beta=beta)
                    
                if np.mod(it, every_t)==0:
                    c_t +=1
                    loss = (output_train[:,it,:]-out_)**2
                    if readout:
                        loss_read = (output_train[:,it,:].dot(readout_wout) - out_.dot(readout_wout))**2
                    
                    #loss_read_ += np.mean(loss[:,0])
                    if readout:
                        loss_kno_ += np.mean(loss_read)                  
                    elif Nkno==0:  
                        loss_kno_ = np.nan(loss_read)
                    else:
                        loss_kno_  += np.mean(loss[:,0:(Nkno)])
                    if Nkno==N:
                        loss_unk_ = np.nan
                        
                    else:
                        loss_unk_ += np.mean(loss[:,Nkno:])
                        
                        
                        
            loss_unk.append(loss_unk_/c_t)
            loss_kno.append(loss_kno_/c_t)
            if cc:
                outsN= out_s.reshape(-1, out_s.shape[-1])
                outtN= output_train.reshape(-1, output_train.shape[-1])
                CC = np.corrcoef(outsN.T, outtN.T)
                # CVs = np.zeros(N)
                CVs = np.diag(CC[0:N,N:])
                loss_kno_R.append(1-np.mean(np.abs(CVs[0:Nkno])))
                if Nkno==N:
                    loss_unk_R.append(np.nan)
                else:
                    loss_unk_R.append(1-np.mean(np.abs(CVs[Nkno:])))
    if calculate:
        return( np.array(loss_unk), np.array(loss_kno), out_s)
    elif cc:
        return( np.array(loss_unk), np.array(loss_kno), np.array(loss_unk_R), np.array(loss_kno_R), )
    else:
        return( np.array(loss_unk), np.array(loss_kno))
    
def get_losses_J(input_train, output_train, Jtrue, taus, net, ws, Nkno, every_ep=10, every_t = 4, calculate=False, greed=False, gain=False, beta=1):

    x0 = net.h0.detach().numpy()
    wi = net.wi.detach().numpy()
    bf = net.b.detach().numpy()[:,0]
    gf = net.g.detach().numpy()[:,0]
    refEI = net.refEI.detach().numpy()
    mwrec = net.mwrec.detach().numpy()
    #Jtrue =  Net.wrec.detach().numpy()
    
    N = net.hidden_size
    loss_unk = []
    loss_unk_greed = []
    loss_kno = []
    loss_J_kno = []
    loss_J_unk = []
    loss_J = []
    
    Jn = Jtrue.dot(np.diag(relu(gf)))
    dta = taus[1]-taus[0]
    for epo in range(np.shape(ws)[-1]):
        if np.mod(epo, every_ep)==0:

            x = x0
            tri = np.shape(output_train)[0]
            x = np.zeros((tri, np.shape(output_train)[-1]))
            print(epo)
            
            out_ = np.zeros((tri, N))
            loss_read_=0
            loss_kno_=0
            loss_unk_=0
            loss_unk_greed_=0
            

            x[0:N,:] = x0
            r = softpl(x+bf)
            c_t = 0
            if calculate and np.mod(epo, 5*every_ep)==0:
                out_s = np.zeros_like(output_train)
                
                # if torch.any(self.refEI==1.):
                #     effW_ = torch.relu(torch.mul(self.wrec, self.refEI))
                #     effW_ = torch.mul(effW_, self.refEI)
                #     effW_ = effW_+torch.mul(-1*self.wrec, torch.abs(self.refEI)-1)
                    
            if np.any(refEI==1.):#refEI[0,0]!=0:
                effW_ = relu(ws[:,:,epo]*refEI)
                effW_ = effW_ *refEI
                effW_ = effW_-ws[:,:,epo]*(np.abs(refEI)-1)
            else:
                effW_ = np.copy(ws[:,:,epo])
            effW = effW_ * mwrec
            
            Jg=effW.dot(np.diag(relu(gf)))
            J_ = effW
            

            out__ = np.zeros_like(output_train)
            for it, ta in enumerate(taus):
                if not gain:
                    out_= r
                else:
                    out_=gf*r
                if calculate:
                    out_s[:,it,:] = out_
                x=x+dta*(-x+(r*relu(gf)).dot(J_.T)+input_train[:,it,:].dot(wi))
                r = softpl(x+bf, beta=beta)
                
                out__[:,it,:] = out_

                Nunk = N-Nkno
                
                Out_map = np.copy(output_train[:,:,Nkno:]) #this is the teacher
                
            if np.mod(epo, 5*every_ep)==0 or epo==np.shape(ws)[-1]-1:
                out__2 = np.copy(out__[:,:,Nkno:])
                lst = np.zeros(Nunk).astype(int)
                
                if greed:
                    counting = np.arange(Nunk)
                    for IIt in range(Nunk):  
                        lNunk = np.shape(out__2)[-1]
                        # Ers = np.zeros((lNunk))
                        # for IIs in range(lNunk):
                        #     Ers[IIs] = np.mean((out__2[:,:,IIs]-Out_map[:,:,IIt])**2)
                        Ers = np.mean(np.mean((out__2-Out_map[:,:,IIt][:,:,np.newaxis])**2, 0),0)
                        lst[IIt]= int(counting[np.argmin(Ers)])
                        mask = np.arange(lNunk)
                        out__2= out__2[:,:, mask!=np.argmin(Ers)]
                        counting = counting[mask!=np.argmin(Ers)]
                        #lNunk = len(counting)
                else:
                    r_a = np.arange(N)
                    lst = r_a[Nkno:]
                
            # for it, ta in enumerate(taus):                    
            #     if np.mod(it, every_t)==0:
            #         out_ = out__[:,it,:]
            #         c_t +=1
            #         loss = (output_train[:,it,:]-out_)**2
            #         loss2 = (output_train[:,it,Nkno:]-out_[:,lst+Nkno])**2
            #         loss_read_ += np.mean(loss[:,0])
            #         if Nkno==0:                    
            #             loss_kno_ = np.nan
            #         else:
            #             loss_kno_  += np.mean(loss[:,0:(Nkno)])
            #         if Nkno==N:
            #             loss_unk_ = np.nan
            #             loss_unk_greed_ = np.nan
            #         else:
            #             loss_unk_ += np.mean(loss[:,Nkno:])
            #             loss_unk_greed_ += np.mean(loss2)
            
            loss = np.mean(np.mean((output_train-out__)**2,0),0)
            loss2 = np.mean(np.mean((output_train[:,:,Nkno:]-out__[:,:,lst+Nkno])**2,0),0)

            loss_kno_ = np.mean(loss[0:Nkno])

            if Nkno==N:
                loss_unk_ = np.nan
                loss_unk_greed_ = np.nan
            else:
                loss_unk_ = np.mean(loss[Nkno:])
                loss_unk_greed_ = np.mean(loss2)

            L_arr = np.arange(N)
            L_arr[N-len(lst):] = lst
            Jg = J_
            Jn = Jtrue
            Jg = Jg[L_arr,:]
            Jg = Jg[:,L_arr]
            
            loss_J.append(np.mean((Jg-Jn)**2))
            kno_neu = np.arange(N-len(lst))
            unk_neu = np.arange(len(lst))+N-len(lst)
            loss_J_kno.append(np.mean((Jg[np.ix_(kno_neu, kno_neu)]-Jn[np.ix_(kno_neu, kno_neu)])**2))
            loss_J_unk.append(np.mean((Jg[np.ix_(unk_neu, unk_neu)]-Jn[np.ix_(unk_neu, unk_neu)])**2))
            
            loss_unk.append(loss_unk_)
            loss_kno.append(loss_kno_)
            loss_unk_greed.append(loss_unk_greed_)

    if calculate:
        return(np.array(loss_unk), np.array(loss_unk_greed), np.array(loss_kno), np.array(loss_J), np.array(loss_J_kno), np.array(loss_J_unk), lst, out_s) #lst: teacher, student
    else:
        return(np.array(loss_unk), np.array(loss_unk_greed), np.array(loss_kno), np.array(loss_J), np.array(loss_J_kno), np.array(loss_J_unk), lst) #lst: teacher, student

