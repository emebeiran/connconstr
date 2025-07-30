import numpy as np
import matplotlib.pyplot as plt
import fun_lib as fl
# This part is for computing Gaussian integrals using the 
# Hermitte polynomials trick
gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def Phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)
def Phi2 (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)
def Prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

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


kaps = np.linspace(0, 1.5, 1000)

fl.set_plot()
#%%

pops = 2
rank = 2

sigma_m2 = np.zeros((rank, pops))
sigma_mn = np.zeros((rank, rank, pops))
gs = np.zeros(pops)
np.random.seed(132)#13

fac =1.
#pop 1
sigma_m2[:,0] = np.array((1., 1.))
sigma_mn[:,:, 0] = np.array(((2., 0.), (0., 0.0)))+0.2*np.random.randn(2,2)#np.array(((2., 0.), (0., 0.6)))#+0.2*np.random.randn(2,2)#np.array(((2.5, 0.9), (-0.9, 0.5)))
gs[0] = 1.2

sigma_m2[:,1] = np.array((1., 1.))
sigma_mn[:,:, 1] = np.array(((0.0, 0), (0., 2.2)))+0.2*np.random.randn(2,2)#np.array( ((0.3, 0.7), (-0.8, 1.5)))
gs[1]=1.5

Sigmas = np.zeros((2*rank, 2*rank, pops))
for ip in range(pops):
    Sigmas[0,0,ip] = sigma_m2[0,ip]
    Sigmas[1,1,ip] = sigma_m2[1,ip]
    Sigmas[0:2,2:, ip] = sigma_mn[:,:,ip]
    Sigmas[2:,0:2, ip] = sigma_mn[:,:,ip].T

    Sigmas[2,2,ip] = 1.
    Sigmas[3,3,ip] = 1.    
    
    e = False
    ct = 0
    while not e:
        u = np.real(np.linalg.eigvals(Sigmas[:,:,ip]))
        if np.min(u)<0:
            e=False
            ct +=1
            Sigmas[2,2,ip] = 1.05*Sigmas[2,2,ip]
            Sigmas[3,3,ip] = 1.05* Sigmas[2,2,ip]
        else:
            e=True
            print(Sigmas[2,2,ip])
M = 0.5*(gs[0]*sigma_mn[:,:,0]+gs[1]*sigma_mn[:,:,1])
Ei = np.linalg.eigvals(M)
print(M)
gts = np.linspace(0.8, 3.)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot([1,1], [-2,2], '--', c='k', lw=1.2)
for igt, gt in enumerate(gts):
    M = 0.5*(gt*sigma_mn[:,:,0]+sigma_mn[:,:,1])
    U, V = np.linalg.eig(M)
    ax1.scatter(np.real(U), np.imag(U), color = 0.2+0.6*np.ones(3)*igt/len(gts))

    V = V/V[0,0]
    v1 = np.real(V[:,0])
    v2 = np.imag(V[:,1])
    
    ax2.plot([0, v1[0]], [0, v1[1]], alpha = 0.2+0.6*igt/len(gts), c='C0', lw=1)
    ax2.plot([0, v2[0]], [0, v2[1]], alpha = 0.2+0.6*igt/len(gts), c='C1', lw=1)
    ax2.scatter(v1[0], v1[1], s=30, c='C0')
    ax2.scatter(v2[0], v2[1], s=30, c='C1')
    
plt.show()
#%%
for x in range(10):
    N = 1000
    X0 = np.random.multivariate_normal(np.zeros(rank*2), Sigmas[:,:,0], size=N)   
    X1 =  np.random.multivariate_normal(np.zeros(rank*2), Sigmas[:,:,1], size=N)  
    
    for ip in range(pops):
        M = X0[:,0:2]
        M[N//2:,:] = X1[N//2:,0:2]
        n = X0[:,2:]
        n[N//2:,:] = X1[N//2:, 2:]
        
        
    u = np.linalg.eigvals(n.T.dot(M)/N)
    plt.scatter(np.real(u), np.imag(u))
    u2 = np.linalg.eigvals(np.mean(sigma_mn,-1))
    plt.scatter(np.real(u2), np.imag(u2))
plt.xlim([0, np.max(np.real(u2))+0.1])
plt.show()
#%%
dt = 0.1
T = 10
time = np.arange(0, T, dt)

reads = np.zeros((len(time), rank))
x0 = 0.2*np.random.randn()*M[:,0]+0.2*np.random.randn()*M[:,1]

Gain = np.ones(N)
Gain[0:N//2] = gs[0]
Gain[N//2:] = gs[1]
reads[0,0] = np.mean(M[:,0]*x0)/np.sqrt(np.mean(M[:,0]**2))
reads[0,1] = np.mean(M[:,1]*x0)/np.sqrt(np.mean(M[:,1]**2))

for it, ti in enumerate(time[:-1]):
    x = x0 + dt*(-x0+(1/N)*M.dot(n.T.dot(Gain*np.tanh(x0))))
    reads[it+1,0] = np.mean(M[:,0]*x)/np.sqrt(np.mean(M[:,0]**2))
    reads[it+1,1] = np.mean(M[:,1]*x)/np.sqrt(np.mean(M[:,1]**2))
    x0 = x
    


#%%
time = np.arange(0, T, dt)
reads2 = np.zeros((len(time), rank))

Gain = np.ones(N)
Gain[0:N//2] = gs[0]
Gain[N//2:] = gs[1]
x0 = np.array((0.5, 0.2))
reads2[0,0] = x0[0]#*(2/3)
reads2[0,1] = x0[1]#*(2/3)

for it, ti in enumerate(time[:-1]):
    k = reads2[it,:]
    kn = k + dt*(-k)
    for ip in range(pops):
        Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
        kn += dt*(gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
    reads2[it+1,0] = kn[0]
    reads2[it+1,1] = kn[1]

#%%
gains1 = np.linspace(0.2, 2.6, 20)
gains2 = np.linspace(0.2, 2.6, 20)

G1s, G2s = np.meshgrid(gains1, gains2)
Loss = np.zeros_like(G1s)
Loss2 = np.zeros_like(G1s)
Loss3 = np.zeros_like(G1s)
Loss4 = np.zeros_like(G1s)
Loss5 = np.zeros_like(G1s)
Loss6 = np.zeros_like(G1s)




GG_ =  np.ones(N)
GG_[0:N//2] = GG_[0:N//2]*gs[0]
GG_[N//2:] = GG_[N//2:]*gs[1]


Gs = np.zeros_like(gs)
for ig1, g1 in enumerate(gains1):
    print(ig1)
    for ig2, g2 in enumerate(gains2):
        G1s[ig1, ig2] = g1
        G2s[ig1, ig2] = g2
        Gs[0] = g1
        Gs[1] = g2
        reads2_ = np.zeros_like(reads2) 
        reads2_[0,:] = reads2[0,:]
        k = reads2_[0,:]        
        for it, ti in enumerate(time[:-1]):
            kn = k + dt*(-k)
            for ip in range(pops):
                Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
                kn += dt*(Gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
            reads2_[it+1,0] = kn[0]
            reads2_[it+1,1] = kn[1]
            k = kn
            
        gg_ = np.ones(N)
        gg_[0:N//2] = gg_[0:N//2]*g1
        gg_[N//2:] = gg_[N//2:]*g2
        
        Loss[ig1, ig2] = np.mean((np.tanh(reads2_.dot(M.T))-np.tanh(reads2.dot(M.T)))**2)
        i2= np.argmin(np.sum((M-np.array((1.5,0)))**2, -1))
        Loss2[ig1, ig2]= np.mean(((np.tanh(reads2_.dot(M.T))[:,i2])-(np.tanh(reads2.dot(M.T)))[:,i2])**2)
        i3= np.argmin(np.sum((M-np.array((0.,1.5)))**2, -1))
        Loss3[ig1, ig2] = np.mean(((np.tanh(reads2_.dot(M.T))[:,i3])-(np.tanh(reads2.dot(M.T)))[:,i3])**2)
        Loss4[ig1, ig2] = np.mean((gg_*np.tanh(reads2_.dot(M.T))-GG_*np.tanh(reads2.dot(M.T)))**2)
        Loss5[ig1, ig2] = np.mean(((gg_[i2]*np.tanh(reads2_.dot(M.T))[:,i2])-(GG_[i2]*np.tanh(reads2.dot(M.T)))[:,i2])**2)
        Loss6[ig1, ig2] = np.mean(((gg_[i3]*np.tanh(reads2_.dot(M.T))[:,i3])-(GG_[i3]*np.tanh(reads2.dot(M.T)))[:,i3])**2)
        
        
        
#%%
Gs[0] = 0.5
Gs[1] = 1.2
reads2_ = np.zeros_like(reads2) 

reads2_[0,:] = x0#reads2[0,:]
reads2[0,:] = x0
k = reads2_[0,:]
kO = reads2[0,:]
for it, ti in enumerate(time[:-1]):
    kn = k + dt*(-k)
    knO = kO-dt*kO
    for ip in range(pops):
        Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
        PriO = Prime(0, np.sum((kO**2*sigma_m2[:,ip])))
        kn += dt*(Gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
        knO += dt*(gs[ip]*PriO*sigma_mn[:,:,ip].T.dot(kO))/pops
    reads2_[it+1,0] = kn[0]
    reads2_[it+1,1] = kn[1]
    reads2[it+1,0] = knO[0]
    reads2[it+1,1] = knO[1]
    k = kn
    kO = knO
fig = plt.figure()
ax = fig.add_subplot(111)
# for i in range(2):
#     plt.plot(time, reads2_[:,i], '--', label=r'$\kappa$'+str(i+1))
plt.plot(time, reads2[:,0], '-k', label=r'$\kappa_1$')
plt.plot(time, reads2[:,1], '--k', label=r'$\kappa_2$')

plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xlabel('time')
ax.set_ylabel('latent $\kappa_i$')
plt.savefig('Figs/gainsTh_ref_traj.pdf')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
# for i in range(2):
#     plt.plot(time, reads2_[:,i], '--', label=r'$\kappa$'+str(i+1))
plt.plot(time, GG_[::50]*np.tanh(reads2.dot(M.T))[:,::50], lw=0.8, c='C0')
plt.plot(time, GG_[i2]*np.tanh(reads2.dot(M.T))[:,i2], '-k', lw=2, label=r'neuron 1')
plt.plot(time, GG_[i3]*np.tanh(reads2.dot(M.T))[:,i3], '--k', lw=2, label=r'neuron 2')
plt.legend(loc=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
ax.set_xlabel('time')
ax.set_ylabel('activity')
plt.savefig('Figs/gainsTh_ref_trajAll_Gain.pdf')
plt.show()


xx1 = np.linspace(-1, 1)
xx2 = np.linspace(-1, 2.2)
XX1, XX2 = np.meshgrid(xx1, xx2)
uu = np.zeros_like(XX1)
vv = np.zeros_like(XX1)
qq = np.zeros_like(XX1)
for ix1, x1 in enumerate(xx1):
    for ix2, x2 in enumerate(xx2):
        k = np.array((x1, x2))
        kn = (-k)
        for ip in range(pops):
            Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
            kn += (gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
        uu[ix1, ix2] = kn[0]
        vv[ix1, ix2] = kn[1]
        qq[ix1, ix2] = np.sum(kn**2)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(reads2[::4,0],reads2[::4,1], '-o', c='k', zorder=10)
#plt.plot(reads2_[::4,0],reads2_[::4,1], '--', c='C1', lw=2)
plt.pcolor(xx1, xx2, np.log10(qq.T), cmap='pink')
plt.streamplot(XX1, XX2, uu.T, vv.T, density=0.7, linewidth=0.8)
plt.plot(reads2[::4,0],reads2[::4,1], '-o', c='k', zorder=10)
ax.set_xlabel('latent $\kappa_1$')
ax.set_ylabel('latent $\kappa_2$')
plt.savefig('Figs/gainsTh_ref_traj2.pdf')
plt.show()



 #%%       
lims = True

points = [[1.5, 2.], [2.1, 1.35],] #[2., 0.6],]
if lims:
    Vmin = -3.
    Vmin2 =Vmin
    Vmax = 0.
    Vmax2=Vmax
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.pcolor(gains1, gains2, np.log10(Loss4), shading='auto', cmap = 'gist_gray', vmin=Vmin, vmax=Vmax)
    plt.contour(gains1, gains2, np.log10(Loss4), colors='k', linestyles='-', levels=[-2.75, -2.25, -1.75, -1.25, -0.75, -0.25])
    
    #cb = plt.colorbar()
    #cb.set_ticks([-3, -2, -1])
    gr = np.gradient(-Loss4)
    plt.streamplot(gains1, gains2, gr[1], gr[0], color='k', density=0.7)
    plt.scatter(gs[1], gs[0], s=50, color='white', zorder=4)
    plt.xlabel(r'gain $g_1$')
    plt.ylabel(r'gain $g_2$')
    #plt.title('2-dim', fontsize=12)
    for ip in range(len(points)):
        PP = points[ip]
        plt.scatter(PP[0], PP[1], marker='s', s=50, zorder=4, edgecolor='w')
    plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels(['-0.5', '1.0', '1.5', '2.0', '2.5'])
    plt.savefig('Figs/gainsTh_lossReal_Gain.pdf', transparent=True)    
    plt.show()

    Vmin = -3.
    Vmin2 =Vmin
    Vmax = 0.
    Vmax2=Vmax
    plt.pcolor(gains1, gains2, np.log10(Loss4), shading='auto', cmap = 'gist_gray', vmin=Vmin, vmax=Vmax)
    
    cb = plt.colorbar()
    cb.set_ticks([ -2, -1, 0])
    gr = np.gradient(-Loss4)
    plt.streamplot(gains1, gains2, gr[1], gr[0], color='k', density=0.7)
    plt.scatter(gs[1], gs[0], s=50, color='white', zorder=4)
    plt.xlabel(r'gain $g_1$')
    plt.ylabel(r'gain $g_2$')
    plt.xlim([-10, -9])
    plt.ylim([-10, -9])
    
    #plt.title('2-dim', fontsize=12)
    for ip in range(len(points)):
        PP = points[ip]
        plt.scatter(PP[0], PP[1], marker='s', s=50, zorder=4, edgecolor='w')
    plt.savefig('Figs/gainsTh_lossReal_cbar_Gain.pdf')    
    plt.show()
    
    
    plt.pcolor(gains1, gains2, np.log10(Loss5), shading='auto', cmap = 'gist_gray', vmin=Vmin, vmax=Vmax)
    plt.contour(gains1, gains2, np.log10(Loss5), colors='k', linestyles='-', levels=[-2.75, -2.25, -1.75, -1.25, -0.75, -0.25])
    
    cb = plt.colorbar()
    cb.set_ticks([ -2, -1, 0])
    plt.scatter(gs[1], gs[0], s=50, color='white', zorder=4)
    gr_0 = np.gradient(-Loss5)
    plt.streamplot(gains1, gains2, gr_0[1], gr_0[0], color='k', density=0.7)
    plt.xlabel(r'gain $g_1$')
    plt.ylabel(r'gain $g_2$')
    #plt.title('$\kappa_1$', fontsize=12)
    for ip in range(len(points)):
        PP = points[ip]
        plt.scatter(PP[0], PP[1], marker='s', s=50, zorder=4, edgecolor='w')
    
    plt.savefig('Figs/gainsTh_lossK1_Gain.pdf')
    plt.show()
    
    
    plt.pcolor(gains1, gains2, np.log10(Loss6), shading='auto', cmap = 'gist_gray', vmin=Vmin2, vmax=Vmax2)
    cb = plt.colorbar()    
    cb.set_ticks([ -2, -1, 0])
    plt.contour(gains1, gains2, np.log10(Loss6), colors='k', linestyles='-', levels=[-2.75, -2.25, -1.75, -1.25, -0.75, -0.25])

    gr_1 = np.gradient(-Loss6)
    plt.streamplot(gains1, gains2, gr_1[1], gr_1[0], color='k', density=0.7)
    plt.scatter(gs[1], gs[0], s=50, color='white', zorder=4)
    plt.xlabel(r'gain $g_1$')
    plt.ylabel(r'gain $g_2$')
    #plt.title('$\kappa_2$', fontsize=12)
    for ip in range(len(points)):
        PP = points[ip]
        plt.scatter(PP[0], PP[1], marker='s', s=50, zorder=4, edgecolor='w')
    plt.savefig('Figs/gainsTh_lossK2_Gain.pdf')

    plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(reads2[::3,0], reads2[::3,1], '-o', c='k', lw=3)
for ip in range(len(points)):
    Gs[0] = points[ip][1]
    Gs[1] = points[ip][0]
    reads2_ = np.zeros_like(reads2) 
    
    reads2_[0,:] = x0#reads2[0,:]
    k = reads2_[0,:]
    kO = reads2[0,:]
    for it, ti in enumerate(time[:-1]):
        kn = k + dt*(-k)
        knO = kO-dt*kO
        for ip in range(pops):
            Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
            kn += dt*(Gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
        reads2_[it+1,0] = kn[0]
        reads2_[it+1,1] = kn[1]

        k = kn
        kO = knO
    
    ax.plot(reads2_[::3,0], reads2_[::3,1], '-o', zorder=5)

plt.pcolor(xx1, xx2, np.log10(qq.T), cmap='pink')

plt.streamplot(XX1, XX2, uu.T, vv.T, density=0.4, linewidth=0.4, color='C0')
ax.set_xlabel(r'latent $\kappa_1$')
ax.set_ylabel(r'latent $\kappa_2$')
ax.set_xlim([-0.5, 1.0])
ax.set_ylim([-0.6, 2.2])

plt.savefig('Figs/gainsTh_ref_fig5E.pdf')
plt.show()



#%%
fig = plt.figure()
ax = fig.add_subplot()


#ax.plot(reads2[::3,0], reads2[::3,1], '-o', c='k', lw=3)
for ip in range(len(points)):
    if ip==0:
        Gs[0] = points[ip][1]
        Gs[1] = points[ip][0]
        reads2_ = np.zeros_like(reads2) 
        
        reads2_[0,:] = x0#reads2[0,:]
        k = reads2_[0,:]
        kO = reads2[0,:]
        for it, ti in enumerate(time[:-1]):
            kn = k + dt*(-k)
            knO = kO-dt*kO
            for ipp in range(pops):
                Pri = Prime(0, np.sum((k**2*sigma_m2[:,ipp])))
                kn += dt*(Gs[ipp]*Pri*sigma_mn[:,:,ipp].T.dot(k))/pops
            reads2_[it+1,0] = kn[0]
            reads2_[it+1,1] = kn[1]
    
            k = kn
            kO = knO
                    
        gg_ = np.ones(N)
        gg_[0:N//2] = gg_[0:N//2]*Gs[0]
        gg_[N//2:] = gg_[N//2:]*Gs[1]
        
        ax.plot(time, GG_[i2]*np.tanh(reads2.dot(M.T))[:,i2], 'k', zorder=5, lw=1.)
        ax.plot(time, gg_[i2]*np.tanh(reads2_.dot(M.T))[:,i2], color='C'+str(ip),lw=2, zorder=5, label='neuron 1')
    
        ax.plot(time, GG_[i3]*np.tanh(reads2.dot(M.T))[:,i3],  '--k',lw=1., zorder=5)
        ax.plot(time, gg_[i3]*np.tanh(reads2_.dot(M.T))[:,i3], '--', color='C'+str(ip),lw=2 , zorder=5, label='neuron 2')
        print(np.mean( (gg_[i3]*np.tanh(reads2_.dot(M.T))[:,i3]-GG_[i3]*np.tanh(reads2.dot(M.T))[:,i3])**2))
ax.set_xlabel(r'time')
ax.set_ylabel(r'activity')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylim([0, 2.2])
plt.legend(frameon=False)
plt.savefig('Figs/gainsTh_ref_fig5F_1_Gain.pdf')
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot()

#points[1] = np.array((0.5, 1.05))
#points2 = [[1.5, 2.], [0.75, 0.85],] 
#ax.plot(reads2[::3,0], reads2[::3,1], '-o', c='k', lw=3)
for ip in range(len(points)):
    if ip==1:
        Gs[0] = points[ip][1]
        Gs[1] = points[ip][0]
        reads2_ = np.zeros_like(reads2) 
        
        reads2_[0,:] = x0#reads2[0,:]
        k = reads2_[0,:]
        kO = reads2[0,:]
        for it, ti in enumerate(time[:-1]):
            kn = k + dt*(-k)
            knO = kO-dt*kO
            for ip in range(pops):
                Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
                kn += dt*(Gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
            reads2_[it+1,0] = kn[0]
            reads2_[it+1,1] = kn[1]
    
            k = kn
            kO = knO
                            
        gg_ = np.ones(N)
        gg_[0:N//2] = gg_[0:N//2]*Gs[0]
        gg_[N//2:] = gg_[N//2:]*Gs[1]
        
        ax.plot(time, GG_[i2]*np.tanh(reads2.dot(M.T))[:,i2], 'k', zorder=5, lw=1.)
        ax.plot(time, gg_[i2]*np.tanh(reads2_.dot(M.T))[:,i2], color='C'+str(ip),lw=2, zorder=5, label='neuron 1')
    
        ax.plot(time, GG_[i3]*np.tanh(reads2.dot(M.T))[:,i3],  '--k',lw=1., zorder=5)
        ax.plot(time, gg_[i3]*np.tanh(reads2_.dot(M.T))[:,i3], '--', color='C'+str(ip),lw=2 , zorder=5, label='neuron 2')
        print(np.mean( (gg_[i2]*np.tanh(reads2_.dot(M.T))[:,i2]-GG_[i2]*np.tanh(reads2.dot(M.T))[:,i2])**2))
ax.set_xlabel(r'time')
ax.set_ylabel(r'activity')
plt.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylim([0, 2.2])
plt.savefig('Figs/gainsTh_ref_fig5F_2_Gain.pdf')
plt.show()


# #%%
# # Do animation

# gains1 = np.linspace(0.2, 2.6, 20)
# gains2 = np.linspace(0.2, 2.6, 20)

# G1s, G2s = np.meshgrid(gains1, gains2)


# Gs = np.zeros_like(gs)

# thess = np.linspace(0, np.pi, 60)
# for ithe, the in enumerate(thess):
#     Loss_ = np.zeros_like(G1s)
#     LossPhi_ = np.zeros_like(G1s)
    
#     for ig1, g1 in enumerate(gains1):
#         print(ig1)
#         for ig2, g2 in enumerate(gains2):
#             G1s[ig1, ig2] = g1
#             G2s[ig1, ig2] = g2
#             Gs[0] = g1
#             Gs[1] = g2
#             reads2_ = np.zeros_like(reads2) 
#             reads2_[0,:] = reads2[0,:]
#             k = reads2_[0,:]        
#             for it, ti in enumerate(time[:-1]):
#                 kn = k + dt*(-k)
#                 for ip in range(pops):
#                     Pri = Prime(0, np.sum((k**2*sigma_m2[:,ip])))
#                     kn += dt*(Gs[ip]*Pri*sigma_mn[:,:,ip].T.dot(k))/pops
#                 reads2_[it+1,0] = kn[0]
#                 reads2_[it+1,1] = kn[1]
#                 k = kn
#             Loss_[ig1, ig2] = np.mean((reads2_[:,0]*np.cos(the)+reads2_[:,1]*np.sin(the)-(reads2[:,0]*np.cos(the)+reads2[:,1]*np.sin(the)))**2)
#             Fac = 1.5
#             LossPhi_[ig1, ig2] = np.mean((np.tanh(Fac*reads2_[:,0]*np.cos(the)+Fac*reads2_[:,1]*np.sin(the))-np.tanh(Fac*reads2[:,0]*np.cos(the)+Fac*reads2[:,1]*np.sin(the)))**2)
#     plt.figure()
#     plt.pcolor(gains1, gains2, np.log10(Loss_), shading='auto', cmap = 'gist_gray', vmin=-3.5, vmax=Vmax)
#     plt.colorbar()
#     gr = np.gradient(-Loss_)
#     plt.streamplot(gains1, gains2, gr[1], gr[0], color='k', density=0.7)
#     plt.scatter(gs[1], gs[0], s=50, color='white', zorder=4)
#     plt.xlabel(r'gain $g_1$')
#     plt.ylabel(r'gain $g_2$')
#     plt.title(r'$\theta = $'+str(the)[0:3], fontsize=12)

#     if ithe < 10:
#         plt.savefig('Figs/gainsThAn_loss_0'+str(ithe)+'.png', dpi=200)    
#     else:
#         plt.savefig('Figs/gainsThAn_loss_'+str(ithe)+'.png', dpi=200)    
#     plt.show()
    
#     plt.figure()
#     plt.pcolor(gains1, gains2, np.log10(LossPhi_), shading='auto', cmap = 'gist_gray', vmin=-3.5, vmax=Vmax)
#     plt.colorbar()
#     gr = np.gradient(-LossPhi_)
#     plt.streamplot(gains1, gains2, gr[1], gr[0], color='k', density=0.7)
#     plt.scatter(gs[1], gs[0], s=50, color='white', zorder=4)
#     plt.xlabel(r'gain $g_1$')
#     plt.ylabel(r'gain $g_2$')
#     plt.title(r'$\theta = $'+str(the)[0:3], fontsize=12)

#     if ithe < 10:
#         plt.savefig('Figs/gainsThPhi_loss_0'+str(ithe)+'.png', dpi=200)    
#     else:
#         plt.savefig('Figs/gainsThPhi_loss_'+str(ithe)+'.png', dpi=200)    
#     plt.show()
        