import torch
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt

B = 2
savepath = "Data/Figure5/teacher.pt"
exec(open('Data/Figure5/setup.py').read())

Jpp = torch.tensor(initJpp(Jpp0, types))
Jpm = torch.tensor(initJpp(Jpm0, types))

#trainable parameters
bm = torch.nn.Parameter(torch.zeros(M))
bp = torch.nn.Parameter(torch.zeros(N))
gm = torch.nn.Parameter(0.5 + torch.rand(M))
gp = torch.nn.Parameter(0.5 + torch.rand(N))
wsp = torch.nn.Parameter(0.5* torch.randn(S, N))

#create model and optimizer
Nepochs = 5000
lr = np.logspace(-2,-3,Nepochs) #decaying learning rate
optimizer = torch.optim.Adam([gm,gp,bm,bp,wsp], lr=lr[0])

track_loss = np.zeros([int(Nepochs/100),2])
count = 0
plt.figure(figsize=(4,8))

for ei in range(Nepochs):
    #initialize hidden states
    um = torch.zeros(B, M)
    up = torch.zeros(B, N)

    m,p = forwardpass(um,up,dt,taum,taup,gm,gp,bm,bp,wsp,Jpm,Jpp)
       
    #losses 
    loss_targ = torch.sum(torch.pow((m - mtarg), 2))/B
    loss_reg = lam*torch.mean(torch.pow(p,2)) #regularize PMN activity
    loss = loss_targ + loss_reg
    
    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.param_groups[0]['lr'] = lr[ei]

    #bounds on gains
    with torch.no_grad():
        gp.data = torch.clamp(gp,0.5,5)
        gm.data = torch.clamp(gm,0.5,5)
    
    #progress
    if ei % 100 == 0:
        print(f'Epoch {ei}, Loss: {loss.item():.4f}')
        track_loss[count,0] = loss_targ
        track_loss[count,1] = loss_reg 
        plotprogress(track_loss,count,m,p,mtarg)
        count += 1

dosave = True
if dosave:
    torch.save([gm,gp,bm,bp,wsp],savepath)



