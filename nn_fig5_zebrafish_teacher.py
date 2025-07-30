import csv
import pickle
import numpy as np
import pandas
import math
import random
import scipy.optimize
import scipy.interpolate
import scipy.io
import h5py
import sklearn
import sklearn.metrics
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from matplotlib import patches
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits import mplot3d

###

#Few General Functions and Initialization of Matrices

def sorted_eigs(X): #ALK: changed sorting to numpy built in
    y,v = np.linalg.eig(X)
    indsort = np.flip(np.argsort(np.real(y)))
    y = y[indsort]
    v = v[:,indsort]
    return y,v

def final_adjustments(W,cdf):
    # make DOs negative
    W[:,cdf.loc['vest']] = -W[:,cdf.loc['vest']]
    W[:,cdf.loc['MO']] = -W[:,cdf.loc['MO']]

    # set ABD,vSPNs outgoing to 0
    W[:,cdf.loc['abdm']] = 0
    W[:,cdf.loc['abdi']] = 0
    W[:,cdf.loc['vspns']] = 0

    # set IBN outgoing to 0
    W[:,cdf.loc['Ibni']] = 0
    W[:,cdf.loc['Ibnm']] = 0

    # make Axial modules 0
    W[:,cdf.loc['axl']] = 0
    W[cdf.loc['axl'],:] = 0

    W[:,cdf.loc['axlm']] = 0
    W[cdf.loc['axlm'],:] = 0
    return W

def get_scaled_slopes(W,cdf,sf=-1):
    y,v = sorted_eigs(W) # call funciton
    slopes = np.real(v[:,0])
    if sf < 0:
        sf = 2.574 / np.mean(slopes[cdf.loc['integ']])
    return sf*slopes
    
#connMatFile = 'Data/Figure5/goldman_data/ConnMatrixPre_cleaned.mat'
connMatFile = 'Data/Figure5/goldman_data/ConnMatrix_CO_top500_2blocks_gamma038_08062020.mat'
connMat = scipy.io.loadmat(connMatFile)
connMatDict = list(connMat)
connMat = np.float32(connMat[connMatDict[-1]])
N = np.shape(connMat)[0]
print(N)

totalInputFile = 'Data/Figure5/goldman_data/totalInputs_CO_top500_2blocks_gamma038_08062020.mat'
totalInputs = scipy.io.loadmat(totalInputFile)
totalInputsDict = list(totalInputs)
totalInputs = np.int32(totalInputs[totalInputsDict[-1]])
totalInputs = np.ravel(totalInputs)

# load cellIDs
cellIDFile  = 'Data/Figure5/goldman_data/cellIDType_CO_top500_2blocks_gamma038_08062020.mat'
cellIDs = scipy.io.loadmat(cellIDFile)
cellIDFileDict = list(cellIDs)
cellIDs = cellIDs[cellIDFileDict[-1]]
cellIDs_unique = set(cellIDs)

matOrderFile  =  'Data/Figure5/goldman_data/MatOrder_CO_top500_2blocks_gamma038_08062020.mat'
matOrder = scipy.io.loadmat(matOrderFile)
idpos = matOrder['MatOrder_CO_top500_2blocks_gamma038_08062020'][0]
    
# get location of neurons
cellLocations =  np.array([(cellIDs == '_Int_'),(cellIDs == 'Ibn_m'),(cellIDs == 'Ibn_i'),(cellIDs == '_MOs_'),(cellIDs == '_Axlm'), (cellIDs == '_Axl_'), (cellIDs == '_DOs_'),(cellIDs == 'ABD_m'),(cellIDs == 'ABD_i'), (cellIDs == 'vSPNs')])
cellNames = ('integ','Ibnm','Ibni','MO','axlm','axl','vest','abdm','abdi','vspns')
lb_cdf = pandas.DataFrame(cellLocations,cellNames)

#ALK: SWITCHED SBM TO LB
connMat[:,lb_cdf.loc['abdm']] = 0
connMat[:,lb_cdf.loc['abdi']] = 0

# normalize W matrix by total inputs
lb_Wnorm = np.zeros(connMat.shape)
for i in np.arange(connMat.shape[0]):
    if totalInputs[i]>0:
        lb_Wnorm[i,:] = connMat[i,:] / totalInputs[i,None]

# Weight matrix including modA connections
lb_Wall = copy.deepcopy(lb_Wnorm)
lb_Wall[:,lb_cdf.loc['vest']] = -lb_Wall[:,lb_cdf.loc['vest']]
lb_Wall[:,lb_cdf.loc['abdm']] = 0
lb_Wall[:,lb_cdf.loc['abdi']] = 0
lb_Wall[:,lb_cdf.loc['vspns']] = 0

# Weight matrix without modA connections
lb_Wnorm = final_adjustments(lb_Wnorm,lb_cdf)
lb_slopes = get_scaled_slopes(lb_Wnorm,lb_cdf)

### Load Existing Clusterings
# Load previous SBM clusterings of the whole center
f = open('Data/Figure5/goldman_data/flatBlockAssignments.data','rb')
blockDict = pickle.load(f)
f.close()
sbmid = blockDict['1 block SBM']
sbmid8_flat = blockDict['8 block SBM']
sbmid9_flat = blockDict['9 block SBM']
sbmid10_flat = blockDict['10 block SBM']
sbmid11_flat = blockDict['11 block SBM']

# Load previous SBM clusterings of modA
f = open('Data/Figure5/goldman_data/axialBlockAssignments.data','rb')
blockDict = pickle.load(f)
f.close()
ax_sbmid = blockDict['1 block SBM']
ax_sbmid2 = blockDict['2 block SBM']
ax_sbmid3 = blockDict['3 block SBM']
ax_sbmid4 = blockDict['4 block SBM']

f = open('Data/Figure5/goldman_data/louvainModO_sbmBlockAssignments.data','rb')
blockDict = pickle.load(f)
f.close()
#sbmid = blockDict['1 block SBM']
sbmid2 = blockDict['2 block SBM']
sbmid3 = blockDict['3 block SBM']
sbmid4 = blockDict['4 block SBM']
sbmid5 = blockDict['5 block SBM']
sbmid6 = blockDict['6 block SBM']
sbmid7 = blockDict['7 block SBM']
sbmid8 = blockDict['8 block SBM']
sbmid9 = blockDict['9 block SBM']
sbmid10 = blockDict['10 block SBM']
sbmid11 = blockDict['11 block SBM']
sbmid12 = blockDict['12 block SBM']


#%%
def simulate_series(W_in,ymax,v_in,ynew=0.99,tau=0.1):
    W = copy.deepcopy(W_in)
    W = ynew*W/ymax
    N = np.shape(W)[0]
    input_filter = 0.001*np.exp(-np.linspace(0,10,101))
    I1 = np.zeros(7000)
    I1[995:1005] = 1e5
    I1 = np.convolve(I1,input_filter)[0:7000]
    I = np.zeros(21000)
    I[0:7000] = I1[:]
    I[7000:14000] = I1[:]
    I[14000:21000] = I1[:]
    r = np.zeros((21000,N))
    dt = 0.001
    positions = np.ones((36,2))
    positions[12:24,0] = 2
    positions[24:,0] = 3
    responses = np.ones((36,N))
    for i in range(1,21000):
        r[i,:] = r[i-1,:] + dt*(np.dot(W,r[i-1,:]) - r[i-1,:] + I[i-1]*v_in)/tau
    return r


y,v1 = sorted_eigs(lb_Wnorm)
my_v_in1 = 0.1*abs(np.random.randn(N)) + np.real(np.sum(v1[:,0:1],axis=1))# + np.imag(np.sum(v1[:,0:1],axis=1))
my_v_in1 += 1*np.real(np.sum(v1[:,1:3],axis=1))# + np.imag(np.sum(v1[:,1:2],axis=1))
lb_rates = simulate_series(W_in=lb_Wnorm,ymax=np.real(y[0]),v_in=my_v_in1,ynew=0.9,tau=1.0)

norm_rates = copy.deepcopy(lb_rates)
for i in range(N):
    if np.max(abs(norm_rates[:7000,i])) > 0:
        norm_rates[:,i] = norm_rates[:,i] / np.max(abs(norm_rates[:7000,i]))

t = np.linspace(0,21,21000)
print(np.argmax(norm_rates[6900,:]))

plt.figure()
plt.plot(t,lb_rates[:,272])
plt.plot(t,lb_rates[:,273])
plt.plot(t,lb_rates[:,281])
plt.show()

#%%
### Get kvalues from simulation
def simulate_ks(W_in,ymax,ynew=0.99,tau=0.1,v_in=[]):
    W = copy.deepcopy(W_in)
    W = ynew*W/ymax
    N = np.shape(W)[0]
    if np.shape(v_in)[0] < N:
        v_in = abs(np.random.randn(N))
        v_in = v_in / np.linalg.norm(v_in)
    input_filter = 0.001*np.exp(-np.linspace(0,10,101))
    I = np.zeros(7000)
    I[995:1005] = 1e5
    I = np.convolve(I,input_filter)[0:7000]
    r = np.zeros((7000,N))
    dt = 0.001
    positions = np.ones((36,2))
    positions[12:24,0] = 2
    positions[24:,0] = 3
    responses = np.ones((36,N))
    for k in range(3):
        r[0,:] = r[-1,:]
        for i in range(1,7000):
            r[i,:] = r[i-1,:] + dt*(np.dot(W,r[i-1,:]) - r[i-1,:] + I[i-1]*v_in)/tau
            #r = r * (r > 0)
        responses[k*12:(k+1)*12,:] = r[3333::333,:]
    ks = np.zeros(N)
    for i in range(N):
        res = scipy.optimize.lsq_linear(positions,responses[:,i],bounds=([0,-np.inf],[np.inf,np.inf]))
        ks[i] = res.x[0]
    return ks



### Plotting Functions
def compare_khist(slopes,cdfs,labels,fill=True,lw=0.5,spacing=1.0,gap=0):
    ax = plt.gca()
    for i in range(0,5):
        plt.axhline(i,linestyle='--',color='gray',alpha=0.5,linewidth=1)
    plt.axhline(5,linestyle='-',color='gray',alpha=0.5,linewidth=1)
    for i in range(6,10):
        plt.axhline(i,linestyle='--',color='gray',alpha=0.5,linewidth=1)
    plt.axhline(10,linestyle='-',color='gray',alpha=0.5,linewidth=1)
    for i in range(11,15):
        plt.axhline(i,linestyle='--',color='gray',alpha=0.5,linewidth=1)
    xcent = [1*spacing,3*spacing,5*spacing]
    bins = np.linspace(-1.5,100.5,103)
    bw = bins[1]-bins[0]
    ec1 = 'black'
    ec2 = 'green'
    ax.add_patch(Rectangle(((xcent[0]+xcent[1])/2,14.25),0.1,bw,facecolor='gray',edgecolor=ec1,linewidth=lw))
    plt.text((xcent[0]+xcent[1])/2+0.15,14.25,labels[0],fontsize=12)
    ax.add_patch(Rectangle(((xcent[1]+xcent[2])/2,14.25),0.1,bw,facecolor='mediumaquamarine',edgecolor=ec2,linewidth=lw))
    plt.text((xcent[1]+xcent[2])/2+0.15,14.25,labels[1],fontsize=12)
    nint = np.sum(cdfs[0].loc['integ'])
    ntrials = np.shape(slopes[0])[1]
    int_slopes = np.reshape(slopes[0][cdfs[0].loc['integ'],:],(nint*ntrials,))
    threshold1 = np.percentile(int_slopes,1)
    threshold2 = np.percentile(int_slopes,99)
    int_slopes[int_slopes <= threshold1] = -99
    int_slopes[int_slopes >= threshold2] = -99
    hist,be = np.histogram(int_slopes,bins=bins)#,density=True)
    hist = hist / (nint*ntrials - np.sum(int_slopes <= threshold1))
    for i in range(1,17):
        if hist[i] > 0.001:
            ax.add_patch(Rectangle((xcent[0]-gap-hist[i],bins[i]),hist[i],bw,facecolor='gray',edgecolor=ec1,linewidth=lw))
    nint = np.sum(cdfs[1].loc['integ'])
    ntrials = np.shape(slopes[1])[1]
    int_slopes = np.reshape(slopes[1][cdfs[1].loc['integ'],:],(nint*ntrials,))
    threshold1 = np.percentile(int_slopes,1)
    threshold2 = np.percentile(int_slopes,99)
    int_slopes[int_slopes <= threshold1] = -99
    int_slopes[int_slopes >= threshold2] = -99
    hist,be = np.histogram(int_slopes,bins=bins)
    hist = hist / (nint*ntrials - np.sum(int_slopes <= threshold1))
    for i in range(1,17):
        if hist[i] > 0.001:
            ax.add_patch(Rectangle((xcent[0]+gap,bins[i]),hist[i],bw,facecolor='mediumaquamarine',edgecolor=ec2,linewidth=lw))
    
    nvest = np.sum(cdfs[0].loc['vest'])
    ntrials = np.shape(slopes[0])[1]
    vest_slopes = np.reshape(slopes[0][cdfs[0].loc['vest'],:],(nvest*ntrials,))
    threshold1 = np.percentile(vest_slopes,1)
    threshold2 = np.percentile(vest_slopes,99)
    vest_slopes[vest_slopes <= threshold1] = -99
    vest_slopes[vest_slopes >= threshold2] = -99
    hist,be = np.histogram(vest_slopes,bins=bins)
    hist = hist / (nvest*ntrials - np.sum(vest_slopes <= threshold1))
    for i in range(1,17):
        if hist[i] > 0.001:
            ax.add_patch(Rectangle((xcent[1]-gap-hist[i],bins[i]),hist[i],bw,facecolor='gray',edgecolor=ec1,linewidth=lw))
    nvest = np.sum(cdfs[1].loc['vest'])
    ntrials = np.shape(slopes[1])[1]
    vest_slopes = np.reshape(slopes[1][cdfs[1].loc['vest'],:],(nvest*ntrials,))
    threshold1 = np.percentile(vest_slopes,1)
    threshold2 = np.percentile(vest_slopes,99)
    vest_slopes[vest_slopes <= threshold1] = -99
    vest_slopes[vest_slopes >= threshold2] = -99
    hist,be = np.histogram(vest_slopes,bins=bins)
    print(np.sum(vest_slopes <= threshold1))
    hist = hist / (nvest*ntrials - np.sum(vest_slopes <= threshold1))
    for i in range(1,17):
        if hist[i] > 0.001:
            ax.add_patch(Rectangle((xcent[1]+gap,bins[i]),hist[i],bw,facecolor='mediumaquamarine',edgecolor=ec2,linewidth=lw))
    
    nabd = np.sum(cdfs[0].loc['abdm'] | cdfs[0].loc['abdi'])
    ntrials = np.shape(slopes[0])[1]
    abd_slopes = np.reshape(slopes[0][cdfs[0].loc['abdm'] | cdfs[0].loc['abdi'],:],(nabd*ntrials,))
    threshold1 = np.percentile(abd_slopes,1)
    threshold2 = np.percentile(abd_slopes,99)
    abd_slopes[abd_slopes <= threshold1] = -99
    abd_slopes[abd_slopes >= threshold2] = -99
    hist,be = np.histogram(abd_slopes,bins=bins)
    hist1 = hist / (nabd*ntrials - np.sum(abd_slopes <= threshold1))
    nabd = np.sum(cdfs[1].loc['abdm'] | cdfs[1].loc['abdi'])
    ntrials = np.shape(slopes[1])[1]
    abd_slopes = np.reshape(slopes[1][cdfs[1].loc['abdm'] | cdfs[1].loc['abdi'],:],(nabd*ntrials,))
    threshold1 = np.percentile(abd_slopes,1)
    threshold2 = np.percentile(abd_slopes,99)
    abd_slopes[abd_slopes <= threshold1] = -99
    abd_slopes[abd_slopes >= threshold2] = -99
    hist,be = np.histogram(abd_slopes,bins=bins)
    hist2 = hist / (nabd*ntrials - np.sum(abd_slopes <= threshold1))
    for i in range(1,17):
        if hist1[i] > 0.001:
            ax.add_patch(Rectangle((xcent[2]-gap-hist1[i],bins[i]),hist1[i],bw,facecolor='gray',edgecolor=ec1,linewidth=lw))
    for i in range(1,17):
        if hist2[i] > 0.001:
            ax.add_patch(Rectangle((xcent[2]+gap,bins[i]),hist2[i],bw,facecolor='mediumaquamarine',edgecolor=ec2,linewidth=lw))
    plt.xticks(xcent,['VPNI','DO','ABD'])
    plt.xlim(xcent[0]-spacing,xcent[2]+spacing)
    plt.ylim(-0.5,15.5)
    plt.yticks([0,5,10,15])


#%%
#Comparison to Functional Imaging - Slopes
y,v = sorted_eigs(lb_Wnorm)

Nsim = 1 #ALK: previous code ran this multiple times but i think it's deterministic
lb_slopes_sim = np.zeros((N,Nsim))
for i in range(Nsim):
    print(i)
    slopes = simulate_ks(lb_Wnorm,np.real(y[0]),ynew=0.9,tau=1)
    sf = 2.574 / np.mean(slopes[lb_cdf.loc['integ']])
    lb_slopes_sim[:,i] = slopes*sf

#%%##

# df = pandas.read_csv('data/slopesThresh_09102020.csv',header=None)
# do_slopes = scipy.io.loadmat('data/fitCellsBasedOnLocationDO.mat')
# do_slopes = do_slopes['slopes'][do_slopes['var2explain'][:,0] > 1,0]
# Nexp = len(df) + np.shape(do_slopes)[0]
# fi_slopes = np.zeros(Nexp)
# isVPNI = np.zeros(Nexp,dtype=bool)
# isABDM = np.zeros(Nexp,dtype=bool)
# isABDI = np.zeros(Nexp,dtype=bool)
# isDO = np.zeros(Nexp,dtype=bool)
# fi_slopes[:len(df)] = df.iloc[:,7].values
# fi_slopes[len(df):] = do_slopes
# isVPNI[:len(df)] = (df.iloc[:,3].values == 0) & (df.iloc[:,4].values == 0) & (df.iloc[:,5].values == 0)
# isABDM[:len(df)] = (df.iloc[:,3].values == 1)
# isABDI[:len(df)] = (df.iloc[:,4].values == 1)
# isDO[len(df):] = 1
# fi_cdf = pandas.DataFrame(np.array([isVPNI,isDO,isABDM,isABDI]),('integ','vest','abdm','abdi'))
# fi_slopes = fi_slopes * 2.574 / np.mean(fi_slopes[fi_cdf.loc['integ']])
# print(np.sum(isVPNI),np.sum(isDO),np.sum(isABDM),np.sum(isABDI))

# plt.figure(1,(10,8))

# # model slopes from simulation
# compare_khist([lb_slopes_sim,np.reshape(fi_slopes,(Nexp,1))],[lb_cdf,fi_cdf],['simulation\n(random inputs)','functional\nimaging'],spacing=0.75,gap=0.007)
# plt.xlim(0.4,4.2)

#%%
ynew = 0.9
W = lb_Wnorm
ymax = np.real(y[0])
W = ynew*W/ymax

input_filter = 0.001*np.exp(-np.linspace(0,10,101))
I1 = np.zeros(7000)
I1[995:1005] = 1e5
I1 = np.convolve(I1,input_filter)[0:7000]
I = np.zeros(21000)
I[0:7000] = I1[:]
I[7000:14000] = I1[:]
I[14000:21000] = I1[:]

v_in = abs(np.random.randn(N))
v_in = np.copy(my_v_in1)

N = np.shape(W)[0]
lent = len(I)

r = np.zeros((N,lent))
dt = 0.001
for it in range(lent-1):
    r[:,it+1] = r[:,it]+dt*(-r[:,it]+W.dot(r[:,it])+I[it]*v_in)

np.savez('zebrafish.npz', W, I, v_in, dt)

#%%
