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

from locallib import makephi, pnorm, truncExp

#%% PARAMS
np.random.seed(43)

learnRateBoundary = .0001
learnRateSlope = .01

nTrials = int(1e5)

pCatch = 0
ITI = .1
rewMag = 1
stim_set = np.linspace(5,95,19)/100#np.linspace(50,100,6)/100#
b = np.median(stim_set)
stim_set = np.delete(stim_set,np.where(np.isclose(stim_set,b)))
# stim_set = np.array([55, 70, 95])/100
stim_noise = .20

cols = ['isChoiceLeft','isCorrect','isCatch','isRewarded','Reward','stim','abs_stim','percept']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)

mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials) # here to be interpreted as p(Reward)

mySession.loc[:, 'abs_stim'] = abs(mySession.loc[:, 'stim'] - b)
mySession.loc[:, 'percept'] = stim_set[np.argmin(abs(np.tile(stim_set,(nTrials,1))-np.tile((mySession.loc[:,'stim']+np.random.randn(nTrials) * stim_noise).values.reshape(-1,1),(1,len(stim_set)))),axis=1)]
mySession.loc[:,['isChoiceLeft','isCorrect','isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch

#% Initialization
mySession.loc[0, 'boundary'] = 0#np.random.rand()#*100
mySession.loc[0, 'slope'] = 0#np.random.rand()*100

#%% Main Loop

iTrial = 0

while (iTrial+1) < nTrials:
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}, b {:1.2f}, m {:1.2f},".format(iTrial + 1, bHat, mHat))

    # S
    x = mySession.loc[iTrial, 'stim']
    xHat = mySession.loc[iTrial, 'percept']
    bHat = mySession.loc[iTrial, 'boundary']
    mHat = mySession.loc[iTrial, 'slope']

    # A
    cho = mHat * (xHat - bHat) > 0
    mySession.loc[iTrial, 'isChoiceLeft'] = cho
    mySession.loc[iTrial, 'isCorrect'] = cho == (x > b)

    # R
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect']
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = rewMag*r

    # S'

    # A'
    delta_bHat = mHat * (r - expit(mHat * abs(xHat - bHat))) * np.sign(bHat - xHat)
    delta_mHat = abs(xHat - bHat) * (r - expit(mHat * abs(xHat - bHat)))
    # delta_mHat = abs(xHat - bHat) * (1 - expit(mHat * abs(xHat - bHat))) * (r + (1 - r) * (
    #             (-expit(mHat * abs(xHat - bHat)) * (1 - pCatch)) / (
    #                 pCatch + (1 - pCatch) * (1 - expit(mHat * abs(xHat - bHat))))))

    bHat = bHat + learnRateBoundary * delta_bHat
    mHat = max(0,mHat + learnRateSlope * delta_mHat)

    mySession.loc[iTrial + 1, 'boundary'] = bHat
    mySession.loc[iTrial + 1, 'slope'] = mHat

    assert(not np.isnan(mySession.loc[iTrial,:].values.astype(float)).any()), "nan found in {}".format(mySession.columns[np.isnan(mySession.loc[iTrial,:].values.astype(float))].values)

    iTrial += 1

#%% Plotting

mySessionLate = mySession.dropna()#.loc[mySession.index > nTrials/2,:]

hf_learning, ha_learning = plt.subplots(2, 2)

ha_learning[0,0].plot(mySession.isCorrect.rolling(window=1000).mean())
ha_learning[0,0].set_xlabel('trial #')
ha_learning[0,0].set_ylabel('Correct choices \n (running average)')

sns.regplot(x='stim', y='isChoiceLeft', data=mySessionLate, logistic=True, ci=None, ax=ha_learning[0,1],x_bins=stim_set)
xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
ha_learning[0,1].plot(xaxis,expit(mHat*(xaxis-bHat)),color='xkcd:silver',label='agents estimate')
ha_learning[0,1].legend(fontsize='small',frameon=False)

ha_learning[1,0].plot(mySession.boundary)
# ha_learning[1,0].set_ylim(0,1)
ha_learning[1,0].set_xlabel('trial #')
ha_learning[1,0].set_ylabel('bHat \n (decision boundary)')

ha_learning[1,1].plot(mySession.slope)
ha_learning[1,1].set_xlabel('trial #')
ha_learning[1,1].set_ylabel('m \n (psychometric slope)')

hf_learning.show()

#%