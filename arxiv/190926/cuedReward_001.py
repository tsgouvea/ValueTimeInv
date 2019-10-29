# import sys
import os
import pickle

import pandas as pd
import numpy as np
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
import seaborn as sns
import sympy as sy

from locallib import makephi, pnorm, truncExp

#%% PARAMS
np.random.seed(43)

learnRateBoundary = .001
learnRateSlope = .1
learnRateRho = .0001
learnRateBeta = .005
temperature = 0.1
learnRateTheta = .001

nTrials = 2000

pCatch = 0.1
Beta = 1.5
ITI = 2
rewMag = 30
stim_set = np.array([5, 30, 45, 55, 70, 95])/100
stim_noise = .2

cols = ['isChoiceLeft','isCorrect','isCatch','isRewarded','Reward','pReward', 'expectedReward','stim','percept','rho',
        'boundary','slope','beta','waitingTime','feedbackTime','trialDur','phiWaitingTime']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols)

mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials)
mySession.loc[:, 'abs_stim'] = abs(mySession.loc[:, 'stim']-stim_set.mean())
mySession.loc[:, 'percept'] = mySession.stim+np.random.randn(nTrials) * stim_noise
mySession.loc[:, 'feedbackTime'] = np.random.exponential(Beta,nTrials)
mySession.loc[:,['isChoiceLeft','isCorrect','isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch

#%% Initialization
mySession.loc[0, 'rho'] = 4
mySession.loc[0, 'beta'] = 1.5
mySession.loc[0, 'boundary'] = .5
mySession.loc[0, 'slope'] = 1
b = stim_set.mean()

Phi = makephi(m=10, nTimeBins=50, isNormalized=True, maxT=20)

Theta = pd.DataFrame(index=Phi.index,columns=stim_set)

if False: # initialize all Theta_of_x to the same random vector
    temp = np.random.randn(Theta.shape[0]) + 1
    for i in range(Theta.shape[1]): Theta.iloc[:,i] = temp
elif False:
    Theta.loc[:,:] = np.random.randn(Theta.shape[0],Theta.shape[1])
elif True:
    Theta.loc[:,:] = 1
elif False:
    if os.path.isfile('Theta.pickle'):
        with open('Theta.pickle', 'rb') as fhandle:
            Theta = pickle.load(fhandle)
else:
    temp = np.flipud(np.exp(-Theta.index))
    for i in range(Theta.shape[1]): Theta.iloc[:, i] = temp

#%%
hf_learnWT, ha_learnWT = plt.subplots(8,2,sharey=True,sharex=True,figsize=(6,12))
ha_learnWT[ha_learnWT.shape[0]-1,0].set_xlabel('waiting time (s)')
ha_learnWT[int(ha_learnWT.shape[0]/2),0].set_ylabel('xHat \n (perceived stimulus)')
ha_learnWT[0,0].set_title('Q(waitingTime)')
ha_learnWT[ha_learnWT.shape[0]-1,1].set_xlabel('waiting time (s)')
ha_learnWT[0,1].set_title('P(waitingTime)')

# hf_learnWT.suptitle('Q / P (s,a)')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Qsa = Theta.T @ Phi
#%% Main Loop

iTrial = 0
i_learnWT = 0

while (iTrial+1) < nTrials:
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}, m {:1.2f}, b {:1.2f}, beta {:1.2f}, rho {:1.2f}".format(iTrial + 1, mHat, bHat, betaHat,
                                                                                       rhoHat))
    # print(iTrial+1)
    #     # print(Theta)
    # if iTrial in (np.logspace(0,np.log10(nTrials-2),ha_learnWT.shape[0]).astype(int)):
    if iTrial in (np.linspace(0, nTrials-2, ha_learnWT.shape[0]).astype(int)):
        ha_learnWT[i_learnWT, 0].imshow(Theta.T @ Phi,
                                aspect='auto', cmap='Reds',
                                extent=list(np.hstack((Phi.columns[[0, -1]].values,
                                                       Theta.columns[[-1, 0]].values))))
        ha_learnWT[i_learnWT, 1].imshow(softmax((Theta.T @ Phi).values / temperature, axis=1),
                                aspect='auto', cmap='Reds',
                                extent=list(np.hstack((Phi.columns[[0, -1]].values,
                                                       Theta.columns[[-1, 0]].values))))
        # hf_learnWT.show()
        i_learnWT+=1
        # ha_learnWT.shape[0] - 1


    # S
    x = mySession.loc[iTrial, 'stim']
    xHat = mySession.loc[iTrial, 'percept']
    bHat = mySession.loc[iTrial, 'boundary']
    mHat = mySession.loc[iTrial, 'slope']
    betaHat = mySession.loc[iTrial, 'beta']
    rhoHat = mySession.loc[iTrial, 'rho']
    f = mySession.loc[iTrial, 'feedbackTime']
    theta = Theta.loc[:, stim_set[np.argmin(abs(xHat-stim_set))]]
    pRew = expit(mHat * (abs(xHat - bHat))) * (1 - pCatch)
    eRew = pRew * rewMag  # from locallib import grad_l
    mySession.loc[iTrial, 'pReward'] = pRew
    mySession.loc[iTrial, 'expectedReward'] = eRew

    # A
    # temperature = 100 if iTrial < nTrials/2 else 0.1
    c = mHat * (xHat - bHat) > 0
    k_phi = np.random.choice(Phi.columns, 1,p=softmax(Phi.T @ theta / temperature)).item()
    k = max(0, betaHat * np.log(eRew / (betaHat * rhoHat)))

    mySession.loc[iTrial, 'isChoiceLeft'] = c
    mySession.loc[iTrial, 'waitingTime'] = k
    mySession.loc[iTrial, 'phiWaitingTime'] = k_phi
    mySession.loc[iTrial, 'isCorrect'] = c == (x > b)

    # R
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[
        iTrial, 'isCatch'] and k > f
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = rewMag*r
    mySession.loc[iTrial, 'trialDur'] = f if mySession.loc[iTrial, 'isRewarded'] else k
    # mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'trialDur'] + ITI
    tau = mySession.loc[iTrial, 'trialDur'] + 1e-3
    phi = Phi.loc[:, Phi.columns[np.argmin(abs(Phi.columns-k))]]
    # if iTrial > nTrials/2 and not r: break
    # S'

    # A'
    # delta_k = eRew * np.exp(k / -betaHat) / betaHat - rhoHat
    delta_rho = r * rewMag - (rhoHat * (tau + ITI))
    delta_bHat = mHat * (expit(mHat * abs(xHat - bHat)) - r) * np.sign(xHat - bHat)
    # delta_mHat = abs(xHat - bHat) * (r - expit(mHat * abs(xHat - bHat)))
    delta_mHat = abs(xHat - bHat) * (1 - expit(mHat * abs(xHat - bHat))) * (r + (1-r)*((-expit(mHat * abs(xHat - bHat)) * (1-pCatch))/(pCatch+(1-pCatch)*(1- expit(mHat * abs(xHat - bHat))))))
    delta_beta = tau/betaHat**2 * (r*(1-(betaHat/tau)) + (1-r)*(1/(1+((1-pRew)/(pRew*np.exp(-tau/betaHat))))))
    delta_theta = (delta_rho - (phi @ theta)) * phi
    # delta_theta = ((r * rewMag - (rhoHat * tau)) - (phi @ theta)) * phi
    rhoHat = max(0, rhoHat + (1 - (1 - learnRateRho) ** (tau + ITI)) * delta_rho)
    bHat = bHat + learnRateBoundary * delta_bHat
    mHat = mHat + learnRateSlope * delta_mHat
    betaHat = betaHat + learnRateBeta * delta_beta
    theta = theta + learnRateTheta * delta_theta

    mySession.loc[iTrial + 1, 'rho'] = rhoHat
    mySession.loc[iTrial + 1, 'boundary'] = bHat
    mySession.loc[iTrial + 1, 'slope'] = mHat
    mySession.loc[iTrial + 1, 'beta'] = betaHat
    Theta.loc[:, stim_set[np.argmin(abs(xHat-stim_set))]] = theta

    assert(not np.isnan(mySession.loc[iTrial,:].values.astype(float)).any()), "nan found in {}".format(mySession.columns[np.isnan(mySession.loc[iTrial,:].values.astype(float))].values)
    assert (rhoHat >= 0), "rhoHat < 0 (={:0.2f})".format(rhoHat)

    iTrial += 1
    # break

with open('Theta.pickle','wb') as fhandle:
    pickle.dump(Theta,fhandle,-1)

#%% Plotting
hf_learnWT.show()

mySession.loc[:,'prevCorr'] = np.hstack((False,mySession.isCorrect.iloc[:-1]))
mySession.loc[:,'prevShort'] = np.hstack((False,mySession.waitingTime.iloc[:-1]<mySession.waitingTime.median()))
mySession.loc[:,'prevRwd'] = np.hstack((False,mySession.isRewarded.iloc[:-1]))
mySession.loc[1:nTrials, 'prevStim'] = mySession.loc[0:nTrials - 2, 'stim'].values
mySessionLate = mySession.dropna()#.loc[mySession.index > nTrials/2,:]

hf_learning, ha_learning = plt.subplots(2, 2)

ha_learning[0,0].plot(mySession.rho)
ha_learning[0,0].plot(np.divide(np.cumsum(mySession.Reward),np.cumsum(mySession.trialDur+ITI)))
ha_learning[0,0].set_xlabel('trial #')
ha_learning[0,0].set_ylabel('rho \n (avg rwd rate)')

ha_learning[0,1].plot(mySession.beta)
ha_learning[0,1].set_xlabel('trial #')
ha_learning[0,1].set_ylabel('beta \n (mean rwd delay)')

ha_learning[1,0].plot(mySession.boundary)
ha_learning[1,0].set_ylim(0,1)
ha_learning[1,0].set_xlabel('trial #')
ha_learning[1,0].set_ylabel('bHat \n (decision boundary)')

ha_learning[1,1].plot(mySession.slope)
ha_learning[1,1].set_xlabel('trial #')
ha_learning[1,1].set_ylabel('m \n (psychometric slope)')
hf_learning.show()

#%
hf_waitingTime, ha_waitingTime = plt.subplots(2, 2)
sns.regplot(x='stim', y='isChoiceLeft', data=mySessionLate, logistic=True, ci=None, ax=ha_waitingTime[0,0],x_bins=stim_set)
xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
ha_waitingTime[0,0].plot(xaxis,expit(mHat*(xaxis-bHat)),color='xkcd:silver',label='agents estimate')
ha_waitingTime[0,0].legend(fontsize='small',frameon=False)
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_waitingTime[0,1],
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stim_set)
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_waitingTime[0,1],
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=stim_set)
x = mySession.loc[iTrial, 'stim']
xHat = mySession.loc[iTrial, 'percept']
bHat = mySession.loc[iTrial, 'boundary']
mHat = mySession.loc[iTrial, 'slope']
betaHat = mySession.loc[iTrial, 'beta']
rhoHat = mySession.loc[iTrial, 'rho']
f = mySession.loc[iTrial, 'feedbackTime']
stim_axis = np.linspace(stim_set.min(),stim_set.max(),91)
ha_waitingTime[1,0].imshow(Theta.T @ Phi,aspect='auto',cmap='Reds',
               extent=list(np.hstack((Phi.columns[[0,-1]].values,Theta.columns[[-1,0]].values))))
ha_waitingTime[1,0].set_xlabel('waiting time (s)')
ha_waitingTime[1,0].set_ylabel('xHat \n (perceived stimulus)')
ha_waitingTime[1,0].set_title('Q(waitingTime)')
ha_waitingTime[1,1].imshow(softmax((Theta.T @ Phi).values/temperature,axis = 1),aspect='auto',cmap='Reds',
               extent=list(np.hstack((Phi.columns[[0,-1]].values,Theta.columns[[-1,0]].values))))
ha_waitingTime[1,1].set_xlabel('waiting time (s)')
ha_waitingTime[1,1].set_ylabel('xHat \n (perceived stimulus)')
ha_waitingTime[1,1].set_title('P(waitingTime)')
hf_waitingTime.show()

#%
x_bins = np.unique(abs((stim_set-stim_set.mean())))

hf_lakS3, ha_lakS3 = plt.subplots(2, 2)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, label='Correct (after correct)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':'}, label='Error (after correct)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, label='Correct (after error)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':'}, label='Error (after error)')
ha_lakS3[0,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevCorr,:],
            logistic=True, ci=None, ax=ha_lakS3[1,0],color='xkcd:black',
            x_bins=x_bins,label='After correct')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevCorr),:],
            logistic=True, ci=None, ax=ha_lakS3[1,0],color='xkcd:gray',
            x_bins=x_bins,label='After error')
ha_lakS3[1,0].legend(frameon=False,fontsize='x-small')


sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, label='Correct (after short)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':'}, label='Error (after short)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, label='Correct (after long)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':'}, label='Error (after long)')
ha_lakS3[0,1].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevShort,:],
            logistic=True, ci=None, ax=ha_lakS3[1,1],color='xkcd:black',
            x_bins=x_bins,label='After short')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevShort),:],
            logistic=True, ci=None, ax=ha_lakS3[1,1],color='xkcd:gray',
            x_bins=x_bins,label='After long')
ha_lakS3[1,1].legend(frameon=False,fontsize='x-small')
hf_lakS3.show()

#%
g = sns.lmplot(x="stim", y="isChoiceLeft", hue="prevStim", data=mySessionLate.loc[mySessionLate.prevRwd, :], y_jitter=0.01,
               palette=sns.color_palette('RdYlBu', 8), fit_reg=True, logistic=True, scatter=False, ci=None)
g.fig.show()


# ha[2].plot(skp.normalize((Phi.T @ Theta).values.T,norm='l1').T)

# plt.figure(figsize=(4, 6))
# plt.subplot(211)
# plt.imshow(Phi, aspect='auto')
# # plt.colorbar()
# plt.subplot(212)
# plt.plot(Phi.T, alpha=.8)

