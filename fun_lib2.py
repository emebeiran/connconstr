#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:41:46 2022

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
    return()

#%%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, b_init, g_init, tM_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_wout = True, train_tau = False, noise_std=0., alpha=0.05, linear=False, relu=True, softplus=False):
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
        elif relu==True:
            self.non_linearity = torch.relu
        else:
            self.non_linearity = torch.tanh
        if softplus==True:
            self.non_linearity = torch.nn.Softplus()
            
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.wi.requires_grad= False
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        self.h0.requires_grad = False
        
        self.tM = nn.Parameter(torch.Tensor(hidden_size))
        if not train_tau:
            self.tM.requires_grad = False
        else:
            self.tM.requires_grad = True
        
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
            self.tM.copy_(tM_init)
            
            if h0_init is None:
                self.h0.zero_
                self.h0.fill_(1)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0)#+self.b.T)
        
        #xB[it+1,:]=xB[it,:]+(dta/tMs)*(-xB[it,:]+np.diag(gf).dot(J.T.dot(relu(xB[it,:])))   + bf + s[it,tr,:].dot(wI))
        #output_train[tr,it+1,:] = wout.T.dot(relu(x0))
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = r.matmul(self.wout)
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * torch.mul(torch.pow(self.tM,-1),(-h + torch.mul(self.g.T, r.matmul(self.wrec.T.t()))+ self.b.T+ input[:,i,:].matmul(self.wi)))
            r = self.non_linearity(h)
            output[:,i+1,:] = r.matmul(self.wout)
        
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

    
def loss_mse_larva(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    
    # Compute loss for each (trial, timestep) (average accross output dimensions)    
    
    loss_tensor = (target[mask>0] - output[mask>0]).pow(2).sum()/(target.shape[0]*target.shape[2]*mask.mean())#(mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    #loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_tensor
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

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=2, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, keep_best=False, cuda=False, save_loss=False, save_params=True, verbose=True, adam=True):
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
    bs = np.zeros((hidden_size, n_epochs))
    gs = np.zeros((hidden_size, n_epochs))
    taus = np.zeros((hidden_size, n_epochs))
    
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

        if np.mod(epoch, 50)==0 and verbose is True:
            if keep_best and np.mean(losses) < best_loss:
                best = net.clone()
                best_loss = np.mean(losses)
                print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
            else:
                print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if torch.cuda.is_available():
            bs[:,epoch]  = net.cpu().b.detach().numpy()[:,0]
            gs[:,epoch]  = net.cpu().g.detach().numpy()[:,0]
            taus[:,epoch] = net.cpu().tM.detach().numpy()[:,0]
            #wr[:,:,epoch] = net.cpu().wrec.detach().numpy()      
            net.to(device=device)
        else:
            bs[:,epoch]  = net.b.detach().numpy()[:,0]
            gs[:,epoch]  = net.g.detach().numpy()[:,0]
            taus[:,epoch] = net.tM.detach().numpy()#[:,0]
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
        return(all_losses, bs, gs, taus, output, output0)
    elif save_loss==False and save_params==True:
        return(bs, gs, output, output0)
    elif save_loss==True and save_params==False:
        return(all_losses, output, output0)
    else:
        return()

def train_larva(net, _input, _target, _mask, n_epochs, lamda=1, lr=1e-2, batch_size=2, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, keep_best=False, cuda=False, save_loss=False, save_params=True, verbose=True, adam=True):
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
    lr=1e-2
    lrs = np.logspace(-2, -3, n_epochs)
    if adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)#
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    hidden_size = net.hidden_size
    bs = np.zeros((hidden_size, n_epochs))
    gs = np.zeros((hidden_size, n_epochs))
    taus = np.zeros((hidden_size, n_epochs))
    
    # graD = np.zeros((hidden_size, n_epochs))
    
    #wr = np.zeros((hidden_size, hidden_size, n_epochs))

    
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
        initial_loss = loss_mse_larva(net(input), target, mask)
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
        loss_ms = loss_mse_larva(output, target[random_batch_idx], mask[random_batch_idx])
        loss_reg = torch.mean(output.pow(2))
        # if epoch<500 and epoch%10==0:
        #     print([loss_ms, loss_reg])
        loss = loss_ms+lamda*loss_reg
        losses.append(loss.item())
        all_losses.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.param_groups[0]['lr'] = lrs[epoch]
        
        with torch.no_grad():
            net.g.data = torch.clamp(net.g, 0.5, None)#Parameter(torch.Tensor(hidden_size,1))
            #gp.data = torch.clamp(gp,0.5,None)
        # These 2 lines important to prevent memory leaks
        # loss.detach_()
        # output.detach_()

        if np.mod(epoch, 50)==0 and verbose is True:
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

    if keep_best:
        net.load_state_dict(best.state_dict())
    if save_loss==True and save_params==True:
        return(all_losses, bs, gs, taus, output, output0)
    elif save_loss==False and save_params==True:
        return(bs, gs, output, output0)
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
def relu(x):
    return x*(x>0)

def softplus(x):
    return(np.log(1+np.exp(x)))

def get_losses(input_train, output_train, taus, net, gs, Nkno, inf_loss, every_ep=1,every_t=4, cpu=False, relu_=True, softplus=True):
    if not cpu:
        Net = net
    else:
        Net = net.cpu()
    x0 = Net.h0.detach().numpy()
    J = Net.wrec.detach().numpy()
    wi = Net.wi.detach().numpy()
    wout = Net.wout.detach().numpy()[:,0]
    bf = Net.b.detach().numpy()[:,0]
    tM = Net.tM.detach().numpy()
    

    trials = np.shape(input_train)[0]
    N = Net.hidden_size
    loss_read  = []
    loss_unk = []
    loss_kno = []
    dta = taus[1]-taus[0]
    for epo in range(np.shape(gs)[-1]):
        if np.mod(epo, every_ep)==0:
            Jg=J.dot(np.diag(gs[:,epo]))
            x = x0
            tri = np.shape(output_train)[0]
            x = np.zeros((tri, N))
            print(epo)
            

            loss_read_=0
            loss_kno_=0
            loss_unk_=0

            x[:,0:N] = x0
            if softplus:
                r = softplus(x)
            else:
                r=relu(x)

            c_t = 0
            # h = h + self.noise_std*noise[:,i,:]+self.alpha * torch.mul(torch.pow(self.tM,-1),(-h + torch.mul(self.g.T, r.matmul(self.wrec.T.t()))+ self.b.T+ input[:,i,:].matmul(self.wi)))
            # r = self.non_linearity(h)
            # xB[it+1,:]=xB[it,:]+(dta/tMs)*(-xB[it,:]+np.diag(gt).dot(J.T.dot(relu(xB[it,:])))   + bf + s[it,tr,:].dot(wI))
        
            for it, ta in enumerate(taus):
                out_= r

                x=x+dta*(1/tM)*(-x+np.dot(r.dot(J),np.diag(gs[:,epo]))+bf+input_train[:,it,:].dot(wi))
                if softplus:
                    r = softplus(x)
                else:
                    r=relu(x)

                if np.mod(it, every_t)==0:
                    c_t +=1
                    
                    
                    loss = (output_train[:,it,:]-out_)**2
                    
                    if inf_loss:
                        loss_kno_  += np.mean(loss[:,0:(Nkno)])
                        if Nkno==N:
                            loss_unk_ = np.nan
                        else:
                            loss_unk_ += np.mean(loss[:,Nkno:])
                    else:
                        loss_kno_  += np.mean(loss[:,-Nkno:])
                        if Nkno==N:
                            loss_unk_ = np.nan
                        else:
                            loss_unk_ += np.mean(loss[:,0:-Nkno])
                        
            loss_unk.append(loss_unk_/c_t)
            loss_kno.append(loss_kno_/c_t)

    return( np.array(loss_unk), np.array(loss_kno))
