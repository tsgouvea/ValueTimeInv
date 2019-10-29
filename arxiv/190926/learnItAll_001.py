import sys
# import os

import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
import seaborn as sns

from locallib import makephi, pnorm, truncExp

sys.version


#%% PARAMS
np.random.seed(43)

learnRateBoundary = .01
learnRateSlope = .01
learnRateRho = .0001
learnRateBeta = .001

nTrials = 3000

pCatch = 0.0
Beta = 1.5
ITI = 0
rewMag = 30
stim_set = np.array([5, 30, 45, 55, 70, 95])/100
stim_noise = .20

cols = ['isChoiceLeft','isCorrect','isCatch','isRewarded','Reward','pReward', 'expectedReward','stim','percept','rho',
        'boundary','slope','beta','waitingTime','feedbackTime','trialDur']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)

mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials)
mySession.loc[:, 'abs_stim'] = abs(mySession.loc[:, 'stim']-stim_set.mean())
mySession.loc[:, 'percept'] = mySession.loc[:, 'stim'] + np.random.randn(nTrials) * stim_noise
mySession.loc[:, 'feedbackTime'] = np.random.exponential(Beta,nTrials)
mySession.loc[:,['isChoiceLeft','isCorrect','isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch

#%% Initialization
mySession.loc[0, 'rho'] = 3
mySession.loc[0, 'beta'] = 1.5
mySession.loc[0, 'boundary'] = .5
mySession.loc[0,'slope'] = 1
b = stim_set.mean()

#%% Main Loop
for iTrial in range(nTrials):
    if (iTrial + 1) % int(nTrials/10) == 0: print(iTrial+1)

    # S
    x = mySession.loc[iTrial, 'stim']
    xHat = mySession.loc[iTrial, 'percept']
    bHat = mySession.loc[iTrial, 'boundary']
    mHat = mySession.loc[iTrial, 'slope']
    betaHat = mySession.loc[iTrial, 'beta']
    rhoHat = mySession.loc[iTrial, 'rho']
    f = mySession.loc[iTrial, 'feedbackTime']
    pRew = expit(mHat * (abs(xHat - bHat))) * (1 - pCatch)  # TODO learn pCatch
    eRew = pRew * rewMag  # from locallib import grad_l
    mySession.loc[iTrial, 'pReward'] = pRew
    mySession.loc[iTrial, 'expectedReward'] = eRew

    # A
    c = mHat * (xHat - bHat) > 0
    k = max(0, betaHat * np.log(eRew / (betaHat * rhoHat)))

    mySession.loc[iTrial, 'isChoiceLeft'] = c
    mySession.loc[iTrial, 'waitingTime'] = k
    mySession.loc[iTrial, 'isCorrect'] = c == (x > b)

    # R
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[
        iTrial, 'isCatch'] and k > f
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = rewMag if mySession.loc[iTrial, 'isRewarded'] else 0
    mySession.loc[iTrial, 'trialDur'] = f if mySession.loc[iTrial, 'isRewarded'] else k
    mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'trialDur'] + ITI
    tau = mySession.loc[iTrial, 'trialDur']

    # S'

    # A'
    delta_k = eRew * np.exp(k / -betaHat) / betaHat - rhoHat
    delta_rho = r * rewMag - (rhoHat * tau)
    delta_bHat = mHat * (expit(mHat * abs(xHat - bHat)) - r) * np.sign(xHat - bHat)
    delta_mHat = abs(xHat - bHat) * (r - expit(mHat * abs(xHat - bHat)))
    # delta_beta = r * (betaHat - k) / (betaHat ** 2)

    rhoHat = max(0, rhoHat + (1 - (1 - learnRateRho) ** tau) * delta_rho)
    bHat = bHat + learnRateBoundary * delta_bHat
    mHat = mHat + learnRateSlope * delta_mHat
    # betaHat = betaHat + learnRateBeta * delta_beta

    if iTrial + 1 == nTrials: break
    mySession.loc[iTrial + 1, 'rho'] = rhoHat
    mySession.loc[iTrial + 1, 'boundary'] = bHat
    mySession.loc[iTrial + 1, 'slope'] = mHat
    mySession.loc[iTrial + 1, 'beta'] = betaHat

    if np.isnan(mySession.loc[iTrial,:].values.astype(float)).any(): break
    if rhoHat < 0 : break
    # break

#%
hf, ha_sign = plt.subplots(2, 2)

ha_sign[0,0].plot(mySession.rho)
ha_sign[0,0].set_xlabel('trial #')
ha_sign[0,0].set_ylabel('rho \n (avg rwd rate)')

ha_sign[0,1].plot(mySession.beta)
ha_sign[0,1].set_xlabel('trial #')
ha_sign[0,1].set_ylabel('beta \n (mean rwd delay)')

ha_sign[1,0].plot(mySession.boundary)
ha_sign[1,0].set_xlabel('trial #')
ha_sign[1,0].set_ylabel('bHat \n (decision boundary)')

ha_sign[1,1].plot(mySession.slope)
ha_sign[1,1].set_xlabel('trial #')
ha_sign[1,1].set_ylabel('m \n (psychometric slope)')

plt.show()


hf, ha = plt.subplots(2, 2)
sns.regplot(x='stim', y='isChoiceLeft', data=mySession, logistic=True, ci=None, ax=ha[0,0],x_bins=stim_set)
xaxis = np.linspace(stim_set.min(),stim_set.max(),100)

ha[0,0].plot(xaxis,expit(mHat*(xaxis-bHat)),color='xkcd:silver',label='agents estimate')
ha[0,0].legend(fontsize='small',frameon=False)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[mySession.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stim_set)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[np.logical_not(mySession.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=stim_set)
# ha[1,0].plot(stim_set,np.clip(K,0,8))


x = mySession.loc[iTrial, 'stim']
xHat = mySession.loc[iTrial, 'percept']
bHat = mySession.loc[iTrial, 'boundary']
mHat = mySession.loc[iTrial, 'slope']
betaHat = mySession.loc[iTrial, 'beta']
rhoHat = mySession.loc[iTrial, 'rho']
f = mySession.loc[iTrial, 'feedbackTime']

stim_axis = np.linspace(stim_set.min(),stim_set.max(),91)
# c = xHat > bHat

K = betaHat*np.log(expit(mHat*(abs(stim_axis-bHat))) * (1-pCatch)*rewMag/(betaHat*rhoHat))
# K[K<0]=0
ha[1,0].plot(stim_axis,K)
ha[1,0].set_xlabel('percept')
ha[1,0].set_ylabel('waiting time')

ha[1,1].plot(mySession.rho)
ha[1,1].set_xlabel('trial #')
ha[1,1].set_ylabel('avg rwd rate')
plt.show()

#%
x_bins = np.unique(abs((stim_set-stim_set.mean())))

mySession.loc[:,'prevCorr'] = np.hstack((False,mySession.isCorrect.iloc[:-1]))
mySession.loc[:,'prevShort'] = np.hstack((False,mySession.waitingTime.iloc[:-1]<mySession.waitingTime.median()))

hf, ha = plt.subplots(2, 2)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,0],
            data=mySession.loc[np.logical_and(mySession.prevCorr,mySession.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, label='Correct (after correct)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,0],
            data=mySession.loc[np.logical_and(mySession.prevCorr,np.logical_not(mySession.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':'}, label='Error (after correct)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,0],
            data=mySession.loc[np.logical_and(np.logical_not(mySession.prevCorr),mySession.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, label='Correct (after error)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,0],
            data=mySession.loc[np.logical_and(np.logical_not(mySession.prevCorr),np.logical_not(mySession.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':'}, label='Error (after error)')
ha[0,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySession.loc[mySession.prevCorr,:],
            logistic=True, ci=None, ax=ha[1,0],color='xkcd:black',
            x_bins=x_bins,label='After correct')
sns.regplot(x='abs_stim', y='isCorrect', data=mySession.loc[np.logical_not(mySession.prevCorr),:],
            logistic=True, ci=None, ax=ha[1,0],color='xkcd:gray',
            x_bins=x_bins,label='After error')
ha[1,0].legend(frameon=False,fontsize='x-small')


sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[np.logical_and(mySession.prevShort,mySession.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, label='Correct (after short)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[np.logical_and(mySession.prevShort,np.logical_not(mySession.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':'}, label='Error (after short)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[np.logical_and(np.logical_not(mySession.prevShort),mySession.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, label='Correct (after long)')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[np.logical_and(np.logical_not(mySession.prevShort),np.logical_not(mySession.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':'}, label='Error (after long)')
ha[0,1].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySession.loc[mySession.prevShort,:],
            logistic=True, ci=None, ax=ha[1,1],color='xkcd:black',
            x_bins=x_bins,label='After short')
sns.regplot(x='abs_stim', y='isCorrect', data=mySession.loc[np.logical_not(mySession.prevShort),:],
            logistic=True, ci=None, ax=ha[1,1],color='xkcd:gray',
            x_bins=x_bins,label='After long')
ha[1,1].legend(frameon=False,fontsize='x-small')
plt.show()

#%

mySession.loc[:,'prevRwd'] = np.hstack((False,mySession.isRewarded.iloc[:-1]))
mySession.loc[1:nTrials, 'prevStim'] = mySession.loc[0:nTrials - 2, 'stim'].values
g = sns.lmplot(x="stim", y="isChoiceLeft", hue="prevStim", data=mySession.loc[mySession.prevRwd, :], y_jitter=0.01,
               palette=sns.color_palette('RdYlBu', 8), fit_reg=True, logistic=True, scatter=False, ci=None)
g.fig.show()

#%% SIGNATURES

mySession.dtypes

hf_sign, ha_sign = plt.subplots(1, 3,figsize=(8,3))

sns.regplot(x='waitingTime', y='isCorrect', data=mySession, logistic=False, ci=None, truncate=True,
            ax=ha_sign[0],x_bins=np.linspace(mySession.waitingTime.min(),mySession.waitingTime.max(),8))


# ha_sign[0].plot(xaxis,expit(mHat*(xaxis-bHat)),color='xkcd:silver',label='agents estimate')
# ha_sign[0].legend(fontsize='small',frameon=False)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_sign[1],
            data=mySession.loc[mySession.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stim_set)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_sign[1],
            data=mySession.loc[np.logical_not(mySession.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=stim_set)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_sign[1],
            data=mySession.loc[mySession.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stim_set)


#%
ndx = mySession.isRewarded
ndx.loc[:] = False

for i, stim in enumerate(mySession.stim.drop_duplicates()):
    med = mySession.waitingTime.loc[mySession.stim==stim].median()
    ndx.loc[np.logical_and(mySession.waitingTime > med,
                           mySession.stim==stim)] = True

sns.regplot(x='abs_stim', y='isCorrect', data=mySession.loc[ndx], logistic=True, ci=None, truncate=True,
            ax=ha_sign[2],x_bins=abs(stim_set - b),
            color='xkcd:dark blue')

sns.regplot(x='abs_stim', y='isCorrect', data=mySession.loc[np.logical_not(ndx)], logistic=True, ci=None, truncate=True,
            ax=ha_sign[2],x_bins=abs(stim_set - b),
            color='xkcd:light blue')

ha_sign[2].set_xlim(0,0.5)
hf_sign.show()