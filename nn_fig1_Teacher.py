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
#%%
N = 300
dta = 0.05
taus = np.arange(0, 20, dta)
Nt = len(taus)
Tau = 0.2

stri = '' 

trials = 60

input_train = np.zeros((trials, Nt, 2))
output_train = np.zeros((trials, Nt, 1))
cond_train = np.zeros((trials))

omega = 3
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
    #%%
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
class RNNgain(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, mask_wrec, refEI, b_init, g_init, h0_init = None,
                 train_b = True, train_g=True, train_conn = False, train_wout = True, noise_std=0., alpha=0.2, linear=False):
        
        super(RNNgain, self).__init__()
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
            self.non_linearity = torch.nn.functional.softplus#torch.tanh
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
        r = self.non_linearity(self.h0+self.b.T)
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = r.matmul(self.wout)
        # clip_weights
        effW_ = torch.relu(torch.mul(self.wrec, self.refEI))
        effW_ = torch.mul(effW_, self.refEI)
        effW = torch.mul(effW_, self.mwrec)
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + torch.mul(torch.relu(self.g.T), r).matmul(effW.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h+self.b.T)
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
input_size = 2
hidden_size = N
output_size = 1
g0 = 1.3
x_init = np.random.randn(input_size, hidden_size)/dta
seed =21
sig_b = 0.1
sig_g = 0.4

np.random.seed(seed)
J = g0*(1/np.sqrt(N))*np.random.randn(N,N)
bt = sig_b*(2*np.random.rand(N)-1.)
gt = sig_g*np.random.randn(N)+1

val = False
while val==False:
    gt[gt<0] = sig_g*np.random.randn(np.sum(gt<0))+1
    val = np.min(gt>0)

def relu(x):
    return(x*(x>0))
gt = relu(gt)


dtype = torch.FloatTensor  
wi_init = torch.from_numpy(x_init).type(dtype)
wwoo = np.zeros((hidden_size, output_size))
wwoo[0]=1.
wo_init = 5*torch.from_numpy(np.random.randn(hidden_size, output_size)/N).type(dtype) #should be N!!

Wr_ini = g0*(1/np.sqrt(N))*np.random.randn(N,N)
wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
wrec_init_Tru = torch.from_numpy(J).type(dtype)

b_init_Tru = torch.from_numpy(bt[:,np.newaxis]).type(dtype)
g_init_Tru = torch.from_numpy(gt[:,np.newaxis]).type(dtype)


b_init_Ran = torch.from_numpy(sig_b*(2*np.random.rand(N,1)-1.)).type(dtype)
g_init_Ran = torch.from_numpy(sig_g*np.random.randn(N,1)+1).type(dtype)
alpha = dta

#%%
val = 10
while val>1:
    print('try')
    g0 = 2.4
    perc = 0.5
    Mask_w = np.random.rand(N,N)>perc
    Wr_ini = g0*(1/np.sqrt(N))*np.random.randn(N,N)
    M_wr = np.ones((N,N))
    percE = 0.7
    M_wr[:,int(percE*N):] = M_wr[:,int(percE*N):]
    
    Wr_ini = Wr_ini+M_wr/N
    Wr_ini = Wr_ini*Mask_w
    Wr_ini[:,int(percE*N):] = 4.0*Wr_ini[:,int(percE*N):]
    m_exc = Wr_ini[:,0:int(percE*N)]<0
    Wr_ini[:,0:int(percE*N)][m_exc] = 0. 
    m_inh = Wr_ini[:,int(percE*N):]>0
    Wr_ini[:,int(percE*N):][m_inh] = 0. 

    wrec_init_Ran = torch.from_numpy(Wr_ini).type(dtype)
    mwrec = torch.from_numpy(Mask_w).type(dtype)
    
    wrec_init_Tru = torch.from_numpy(J).type(dtype)
    
    refEI_ = np.ones_like(Wr_ini)
    refEI_ = refEI_.dot(np.diag(np.sign(np.sum(Wr_ini,0))))
    refEI = torch.from_numpy(refEI_).type(dtype)
    
    
    x = np.random.randn(N)
    T = 20
    dt = 0.1
    tt = np.arange(0, T, dt)
    
    xs = np.zeros((len(tt), N))
    x0 = np.random.randn(N)
    xs[0,:] = x0
    
    def softpl(x):
        return (np.log(1+np.exp(x)))
    
    def d_soft(x):
        return (np.exp(x)/(1+np.exp(x)))
    
    
    for it, ti in enumerate(tt[:-1]):
        xs[it+1,:] = xs[it,:]+dt*(-xs[it,:] + Wr_ini.dot(gt*softpl(xs[it,:]+bt)))
    
    Jac = Wr_ini.dot(np.diag(gt*d_soft(xs[-1,:]+bt)))
    
    ev = np.linalg.eigvals(Jac)
    plt.scatter(np.real(ev), np.imag(ev))
    plt.show()
    val = np.max(np.real(ev))
    plt.figure()
    plt.hist(xs[-1,:])
    plt.show()
    plt.figure()
    plt.plot(xs[:,0:3])
    plt.show()
    
    plt.figure()
    plt.hist(softpl(xs[-1,:]+bt), 50)
    plt.show()
    H0 =torch.from_numpy(xs[-1,:]).type(dtype)
#%%

# try:
#     AA = np.load(stri+'example_net_N_'+str(N)+'_2.npz') #, J, gf, bf, wI, wout, x0
#     J = AA['arr_0']
#     gf = AA['arr_1']
#     bf = AA['arr_2']
#     wI = AA['arr_3']
#     wout = AA['arr_4']
#     x0 = AA['arr_5']
# except:
    
NetJ = RNNgain(input_size, hidden_size, output_size, wi_init, wo_init, wrec_init_Ran, mwrec,refEI, b_init_Tru, g_init_Tru, h0_init=H0,
                          train_b = True, train_g=True, train_conn = True, noise_std=0.01, alpha=alpha, linear=False)


mask_train = np.ones((trials, Nt, N))

input_Train = torch.from_numpy(input_train).type(dtype)
output_Train = torch.from_numpy(output_train).type(dtype) 
mask_train = torch.from_numpy(mask_train).type(dtype) 



outt = NetJ(input_Train)
outt = outt.detach().numpy()
plt.plot(outt[35,:,0].T)
plt.plot(outt[15,:,0].T)
plt.show()

#%%
do_times = 7#7
for do in range(do_times):
    n_ep = 200
    lr = 0.001#0.005
    
    lossJ, bsJ, gsJ, wsJ, outJ, outJ_ = train(NetJ, input_Train, output_Train, mask_train, n_epochs=n_ep, plot_learning_curve=True, plot_gradient=True, 
                              lr=lr, clip_gradient = 5., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True)

    outt = NetJ(input_Train)
    outt = outt.detach().numpy()
    plt.plot(outt[35,:,0].T)
    plt.show()
    
    plt.imshow(NetJ.wrec.detach().numpy(), cmap='bwr', vmin=-0.1, vmax=0.1)
    plt.show()
lr_s = 0.001#0.005

# lossJ, bsJ, gsJ, wsJ, outJ, outJ_ = train(NetJ, input_Train, output_Train, mask_train, n_epochs=n_ep, plot_learning_curve=True, plot_gradient=True, 
#                           lr=lr_s, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True, save_params=True, adam=True)

#%%
J = NetJ.wrec.detach().numpy()
J0 = Wr_ini
gf = NetJ.g.detach().numpy()[:,0]
bf = NetJ.b.detach().numpy()[:,0]
wI = NetJ.wi.detach().numpy()
wout = NetJ.wout.detach().numpy()[:,0]
mwrec = NetJ.mwrec.detach().numpy()
refEI = NetJ.refEI.detach().numpy()

g0 = gt
x0 = NetJ.h0.detach().numpy()

#%%
torch.save(NetJ.state_dict(), "netGain_softplus_2.pt")
np.savez('example_netGain_softplus_2.npz', J, gf, bf, wI, wout, x0, mwrec, refEI)


#%%
plt.figure()
outt = NetJ(input_Train)
outt = outt.detach().numpy()
plt.plot(outt[28,:,0].T)
plt.show()
#%%
plt.figure()
plt.imshow(NetJ.wrec.detach().numpy(), cmap='bwr', vmin=-0.1, vmax=0.1)
plt.show()
#%%

input_trainLong = np.zeros((trials, 3*Nt, 2))
input_trainLong[:,0:Nt,:] = np.copy(input_train)
input_TrainLong= torch.from_numpy(input_trainLong).type(dtype)
wo_init2 = torch.from_numpy(np.eye(N)).type(dtype) 

NetJexp = RNNgain(input_size, hidden_size, N, NetJ.wi, wo_init2, NetJ.wrec, NetJ.mwrec, NetJ.refEI, NetJ.b, NetJ.g, 
                          train_b = False, train_g=False, train_conn = True, noise_std=0.001, alpha=alpha, linear=False, h0_init=NetJ.h0)
#%%
#find three indeces
trialsT = np.zeros(3)
count_tr = 0
while np.min(trialsT)==0:
    if input_TrainLong[count_tr,:,:].sum(axis=0).detach().numpy().sum()==0 and trialsT[2]==0:
        trialsT[2] = count_tr
    elif input_TrainLong[count_tr,:,:].sum(axis=0).detach().numpy()[0]==1. and trialsT[1]==0:
        trialsT[1] = count_tr
    elif input_TrainLong[count_tr,:,:].sum(axis=0).detach().numpy()[1]==1. and trialsT[0]==0:
        trialsT[0] = count_tr
    count_tr+=1
outt = NetJexp(input_TrainLong[trialsT,:,:]).detach().numpy()

XL = 1000
fig = plt.figure(figsize=[10,5])
ax = fig.add_subplot(131)
plt.imshow(outt[0,:,:].T, vmin=0.1, vmax=1.5)
plt.xlim([0,XL])
ax = fig.add_subplot(132)
plt.imshow(outt[1,:,:].T, vmin=0.1, vmax=1.5)
plt.xlim([0,XL])
ax = fig.add_subplot(133)
plt.imshow(outt[2,:,:].T, vmin=0.1, vmax=1.5)
plt.xlim([0,XL])
#%%
Wout = NetJ.wout.detach().numpy()
fig = plt.figure(figsize=[10,5])
for i in range(3):
    ax = fig.add_subplot(1,3,i+1)
    plt.plot(np.dot(Wout.T, outt[i,:,:].T )[0])
    ax.set_ylim([-1, 1])
         
#%%

g_sol = NetJ.g.detach().numpy()
b_sol = NetJ.b.detach().numpy()
#%%
g_ran = np.random.permutation(g_sol)
b_ran = np.random.permutation(b_sol)
G_ran = 0.5*torch.from_numpy(g_ran).type(dtype)
B_ran = torch.from_numpy(b_ran).type(dtype)
NetJ_gb = RNNgain(input_size, hidden_size, N, NetJ.wi, wo_init, NetJ.wrec, NetJ.mwrec,NetJ.refEI, B_ran, G_ran, 
                          train_b = False, train_g=False, train_conn = True, noise_std=0.001, alpha=alpha, linear=False, h0_init=NetJ.h0)
NetJ_g = RNNgain(input_size, hidden_size, N, NetJ.wi, wo_init, NetJ.wrec, NetJ.mwrec,NetJ.refEI, NetJ.b, G_ran, 
                          train_b = False, train_g=False, train_conn = True, noise_std=0.001, alpha=alpha, linear=False, h0_init=NetJ.h0)
NetJ_b = RNNgain(input_size, hidden_size, N, NetJ.wi, wo_init, NetJ.wrec, NetJ.mwrec,NetJ.refEI, B_ran, NetJ.g, 
                          train_b = False, train_g=False, train_conn = True, noise_std=0.001, alpha=alpha, linear=False, h0_init=NetJ.h0)


#%%

Wr_ini2 = NetJ.wrec.detach().numpy()
permg = True
permb = True
if permg:
    gt2 = 0.5*np.random.permutation(NetJ.g.detach().numpy()[:,0])#np.random.permutation
else:
    gt2 = (NetJ.g.detach().numpy()[:,0])#
if permb:
    bt2 = np.random.permutation(NetJ.b.detach().numpy()[:,0])
else:
    bt2 = (NetJ.b.detach().numpy()[:,0])


#%%
x = np.random.randn(N)
T = 50
dt = 0.1
tt = np.arange(0, T, dt)

xs = np.zeros((len(tt), N))
x0 = np.random.randn(N)
xs[0,:] = x0

ev1 = np.linalg.eigvals(Wr_ini2.dot(np.diag(gt2)))
plt.scatter(np.real(ev1), np.imag(ev1), c='k')

def softpl(x):
    return (np.log(1+np.exp(x)))

def d_soft(x):
    return (np.exp(x)/(1+np.exp(x)))

for it, ti in enumerate(tt[:-1]):
    xs[it+1,:] = xs[it,:]+dt*(-xs[it,:] + Wr_ini2.dot(gt2*softpl(xs[it,:]+bt2)))

# xs[0,:] = xs[-1,:]
# for it, ti in enumerate(tt[:-1]):
#     xs[it+1,:] = xs[it,:]+dt*(-xs[it,:] + Wr_ini2.dot(gt2*softpl(xs[it,:]+bt2)))
    
Jac = Wr_ini2.dot(np.diag(gt2*d_soft(xs[-1,:]+bt2)))

ev = np.linalg.eigvals(Jac)
plt.scatter(np.real(ev), np.imag(ev))

plt.xlim([-2, 2])
plt.show()
val = np.max(np.real(ev))

plt.figure()
plt.plot(softpl(xs[:,0:5]+bt2[0:5]))
plt.show()
