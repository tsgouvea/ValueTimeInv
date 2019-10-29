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

# #%% CRITIC
#
# if 'critic' not in locals():
#     critic = kr.models.Sequential()
#     critic.add(kr.layers.Dense(units=10,activation='softplus',input_dim=3))
#     critic.add(kr.layers.Dense(units=10,activation='softplus'))
#     critic.add(kr.layers.Dense(units=1, activation='linear'))
# critic.compile(optimizer='SGD',
#               loss='mse')
# critic.summary()

#%% TOY DATA

nRuns = 3
nTrials = 10000
pCatch = 0.1
# pCorrect = 0.3

ITI = 10
rewMag = 30
Beta = 1.5

learnBrake = 1#(2**(1/(nTrials-1)))

hf, ha = plt.subplots(nRuns, 3,figsize=(8,2*nRuns),sharey='col')

# for irun, pCorrect in enumerate(np.linspace(.2,.8,nRuns)):
for irun, pCorrect in enumerate([.95, .7, .55]):
    learnRateWait = .01
    learnRateRho = .001
    learnRateBeta = .01

    np.random.seed(42)
    isCorrect = np.random.rand(nTrials) < pCorrect

    mySession = pd.DataFrame({'isCorrect':isCorrect})
    mySession.loc[:,'p_rew'] = pCorrect
    mySession.loc[:,'waitingTime'] = np.nan
    mySession.loc[:,'feedbackTime'] = truncExp(Beta,0.,18.,nTrials).astype(float)
    mySession.loc[:,'isCatch'] = np.random.rand(nTrials) < pCatch
    mySession.loc[:, 'isRewarded'] = np.nan
    mySession.loc[:, 'rho'] = np.nan
    mySession.loc[:, 'beta'] = np.nan

    # X = [mySession.loc[:,var].values.astype(float) for var in ['perc','isChoiceLeft','waitingTime']]

    #%
    # rho = pd.Series(index=np.arange(nTrials))#,columns=np.sort(mySession.istim.drop_duplicates()))
    # rho.loc[0] = 0
    #%
    for iTrial in range(nTrials):
        # S
        # mySession.loc[iTrial, 'feedbackTime'] = truncExp(Beta, .5, 8)
        # A
        mySession.loc[iTrial, 'waitingTime'] = 0 if iTrial == 0 else mySession.loc[iTrial, 'waitingTime']
        mySession.loc[iTrial, 'rho'] = 0 if iTrial == 0 else mySession.loc[iTrial, 'rho']
        mySession.loc[iTrial, 'beta'] = .1 if iTrial == 0 else mySession.loc[iTrial, 'beta']
        # R
        mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[iTrial, 'isCatch'] and mySession.loc[
                iTrial, 'waitingTime'] > mySession.loc[iTrial, 'feedbackTime']
        mySession.loc[iTrial, 'reward'] = rewMag * float(mySession.loc[iTrial, 'isRewarded'])
        mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'waitingTime'] + ITI
        # mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'feedbackTime'] if mySession.loc[
        #     iTrial, 'isRewarded'] else mySession.loc[iTrial, 'waitingTime']

        if iTrial % 500 == 0:
            print(iTrial)
        # S'
        # A'
        deltaBeta = (mySession.loc[iTrial, 'feedbackTime'] - mySession.loc[iTrial, 'beta']) / mySession.loc[
            iTrial, 'beta'] ** 2 if mySession.loc[iTrial, 'isRewarded'] else 0 #from locallib import grad_beta
        mySession.loc[iTrial +1 , 'beta'] = mySession.loc[iTrial, 'beta'] + learnRateBeta*deltaBeta

        delta_k = rewMag * mySession.loc[iTrial, 'p_rew'] * np.exp(
            mySession.loc[iTrial, 'waitingTime'] / -mySession.loc[iTrial, 'beta']) / mySession.loc[iTrial, 'beta'] - \
                  mySession.loc[iTrial, 'rho']
        mySession.loc[iTrial + 1, 'waitingTime'] = max(0,
                                                       mySession.loc[iTrial, 'waitingTime'] + learnRateWait * delta_k)

        delta_rho = mySession.loc[iTrial, 'isRewarded'] - mySession.loc[iTrial, 'rho'] * mySession.loc[
            iTrial, 'trialDur']
        mySession.loc[iTrial + 1, 'rho'] = mySession.loc[iTrial, 'rho'] + (
                1 - (1 - learnRateRho) ** mySession.loc[iTrial, 'trialDur']) * delta_rho

        if iTrial + 1 == nTrials: break
        learnRateWait = learnRateWait**learnBrake
        learnRateRho = learnRateRho**learnBrake
        learnRateBeta = learnRateBeta**learnBrake
    #%


    # blah = np.log(mySession.p_rew.loc[0]/mySession.rho.loc[nTrials-1])
    # label = '$\beta$' if irun == 0 else None
    label = 'incr'# if irun == 0 else None
    ha[irun,0].plot(mySession.loc[:,'waitingTime'],label=label)
    label = 'log({:0.2f}/rho)'.format(pCorrect)# if irun == 0 else None
    ydata = np.multiply(mySession.beta,np.log(rewMag * mySession.p_rew.loc[0]/np.multiply(mySession.beta,mySession.rho)))
    ha[irun,0].plot(np.arange(nTrials)[int(nTrials/2):],ydata[int(nTrials/2):nTrials],label=label)
    ha[irun,0].legend(fontsize='small',frameon=False)

    ha[irun,0].set_xlabel('trial #')
    ha[irun,0].set_ylabel('waitingTime')
    ha[irun, 0].set_title('P(reward) = {:0.2f}'.format(pCorrect))

    ha[irun,1].plot(mySession.loc[:,'rho'])
    ha[irun,1].set_xlabel('trial #')
    ha[irun,1].set_ylabel('rho \n (avg rwd rate)')

    ha[irun, 2].plot(mySession.loc[:, 'beta'])
    ha[irun, 2].set_xlabel('trial #')
    ha[irun, 2].set_ylabel('beta')
    break

plt.tight_layout()
plt.show()