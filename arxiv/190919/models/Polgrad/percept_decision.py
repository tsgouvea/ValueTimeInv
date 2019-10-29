import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
import seaborn as sns
# import tensorflow.keras as kr
from matplotlib import rc
# rc('text', usetex=True)

from locallib import makephi, pnorm, truncExp

sys.version

#%% TOY DATA

nTrials = 1000
pCatch = 0.1
# pCorrect = 0.3
Beta = 1.5
ITI = 2
learnRateWait = .01
learnRateRho = .001
learnRatePerc = .01
rewMag = 30

stim_set = np.array([5, 30, 45, 55, 70, 95])
stim_noise = 20

np.random.seed(42)

cols = ['stim', 'perc', 'z', 'isChoiceLeft','isCorrect','rho','boundary','slope']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols)

mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials)
mySession.loc[:, 'perc'] = mySession.loc[:, 'stim'] + np.random.randn(nTrials) * stim_noise

# isCorrect = np.random.rand(nTrials) < pCorrect
mySession.loc[:,'p_rew'] = np.nan
mySession.loc[:,'waitingTime'] = np.nan
mySession.loc[:,'feedbackTime'] = truncExp(Beta,0.,18.,nTrials).astype(float)
mySession.loc[:,'isCatch'] = np.random.rand(nTrials) < pCatch
mySession.loc[:, 'isRewarded'] = np.nan
mySession.loc[:, 'rho'] = np.nan
mySession.loc[:, 'beta'] = Beta


# X = [mySession.loc[:,var].values.astype(float) for var in ['perc','isChoiceLeft','waitingTime']]

#%
# rho = pd.Series(index=np.arange(nTrials))#,columns=np.sort(mySession.istim.drop_duplicates()))
# rho.loc[0] = 0
#%%
for iTrial in range(nTrials):
    # S
    # mySession.loc[iTrial, 'feedbackTime'] = truncExp(Beta, .5, 8)
    # A
    mySession.loc[iTrial, 'rho'] = 0 if iTrial == 0 else mySession.loc[iTrial, 'rho']
    mySession.loc[iTrial, 'boundary'] = 0 if iTrial == 0 else mySession.loc[iTrial, 'boundary']
    mySession.loc[iTrial, 'slope'] = 0 if iTrial == 0 else mySession.loc[iTrial, 'slope']
    mySession.loc[iTrial,'isChoiceLeft'] = float(mySession.loc[iTrial, 'stim'] > mySession.loc[iTrial, 'boundary'])
    mySession.loc[iTrial, 'waitingTime'] = mySession.beta.loc[iTrial]
    break

    # R
    mySession.loc[iTrial, 'isRewarded'] = rewMag*float(
        mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[iTrial, 'isCatch'] and mySession.loc[
            iTrial, 'waitingTime'] > mySession.loc[iTrial, 'feedbackTime'])
    mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'waitingTime'] + ITI
    # mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'feedbackTime'] if mySession.loc[
    #     iTrial, 'isRewarded'] else mySession.loc[iTrial, 'waitingTime']

    if iTrial % 500 == 0:
        print(iTrial)
    # S'
    # A'
    delta_k = rewMag * mySession.loc[iTrial, 'p_rew'] * np.exp(
        mySession.loc[iTrial, 'waitingTime'] / -Beta) / Beta - \
              mySession.loc[iTrial, 'rho']

    mySession.loc[iTrial + 1, 'waitingTime'] = max(0,mySession.loc[iTrial, 'waitingTime'] + learnRateWait * delta_k)

    delta_rho = mySession.loc[iTrial, 'isRewarded'] - mySession.loc[iTrial, 'rho'] * mySession.loc[
        iTrial, 'trialDur']
    mySession.loc[iTrial + 1, 'rho'] = mySession.loc[iTrial, 'rho'] + (
            1 - (1 - learnRateRho) ** mySession.loc[iTrial, 'trialDur']) * delta_rho

    mySession.loc[n + 1, 'boundary'] = mySession.loc[n, 'boundary'] - learnRatePerc * mySession.loc[n, 'delta'] * mySession.loc[
        n, 'slope']  # (float(mySession.loc[n, 'stim'] > stim_set.mean()) - expit(mySession.loc[n,'perc']-mySession.loc[n,'boundary']))
    mySession.loc[n + 1, 'slope'] = mySession.loc[n, 'slope'] + learnRatePerc * mySession.loc[n, 'z'] * mySession.loc[n, 'delta']

    if iTrial + 1 == nTrials: break
#%%


# blah = np.log(mySession.p_rew.loc[0]/mySession.rho.loc[nTrials-1])
# label = '$\beta$' if irun == 0 else None
label = 'incr'# if irun == 0 else None
ha[irun,0].plot(mySession.loc[:,'waitingTime'],label=label)
label = 'log({:0.2f}/rho)'.format(pCorrect)# if irun == 0 else None
ha[irun,0].plot(np.arange(nTrials)[int(nTrials/2):],np.log(rewMag * mySession.p_rew.loc[0]/mySession.rho)[int(nTrials/2):nTrials],label=label)
ha[irun,0].legend(fontsize='small',frameon=False)

ha[irun,0].set_xlabel('trial #')
ha[irun,0].set_ylabel('waitingTime')
ha[irun, 0].set_title('P(reward) = {:0.2f}'.format(pCorrect))

ha[irun,1].plot(mySession.loc[:,'rho'])
ha[irun,1].set_xlabel('trial #')
ha[irun,1].set_ylabel('rho \n (avg rwd rate)')
# break

plt.tight_layout()
plt.show()