import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import expit

import seaborn as sns

from locallib import optimwt, grad_rho, grad_psyc

#%% PARAMS
# np.random.seed(43)

learnRateRho = .001
learnRateBeta = .01
learnRateL = .01

rho_method='G3'
# rho_method = 'lak14'
# beta_method = 'logprob'
# beta_method = 'prob'

nTrials = int(1e4)

pCatch = 0.1
Beta = 1.5
ITI = 1
m = 30 #reward magnitude
stim_set = np.linspace(5,95,10)/100#
stim_noise = .2

cols = ['isCorrect','isCatch','isRewarded','Reward','waitingTime','feedbackTime','trialDur']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)

mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials) # here to be interpreted as p(Reward)
mySession.loc[:, 'percept'] = mySession.loc[:, 'stim'] + np.random.randn(nTrials)*stim_noise # here to be interpreted as p(Reward)
mySession.loc[:, 'feedbackTime'] = np.random.exponential(Beta,nTrials)
mySession.loc[:,['isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch

#% Initialization
mySession.loc[0, 'rho'] = 8
mySession.loc[0, 'beta'] = Beta
mySession.loc[0, 'pi'] = -np.log(pCatch/(1-pCatch))
mySession.loc[0, 'm'] = 1.6/stim_noise #np.pi/(stim_noise*np.sqrt(3))
mySession.loc[0, 'b'] = 0.5
for param in ['rho','beta','pi','m','b']:
    mySession.loc[0,param] += np.random.randn()*0.1*mySession.loc[0,param]
    print("{}: {:1.2f}".format(param,mySession.loc[0,param]))
    # break

#%% Main Loop

iTrial = 0

while (iTrial+1) < nTrials:
    #%%
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}, rho {:1.2f}, beta {:1.2f}, pi {:1.2f}, m {:1.2f}, b {:1.2f}, alpha {:1.2f}".format(
            iTrial + 1, rhoHat, betaHat, piHat, mHat, bHat,
            expit(mHat * np.abs(xHat - bHat)) * expit(piHat) * (1 - np.exp(-k / betaHat))))

    # S
    xHat = mySession.loc[iTrial, 'percept']
    betaHat = mySession.loc[iTrial, 'beta']
    rhoHat = mySession.loc[iTrial, 'rho']
    piHat = mySession.loc[iTrial, 'pi']
    mHat = mySession.loc[iTrial, 'm']
    bHat = mySession.loc[iTrial, 'b']
    q = expit(mHat * np.abs(xHat - bHat)) * expit(piHat)
    d = mySession.loc[iTrial, 'feedbackTime']

    # A
    # temperature = 100 if iTrial < nTrials/2 else 0.1
    k = optimwt(beta=betaHat,rho=rhoHat,q=q,m=m,method=rho_method)
    a = xHat > bHat
    mySession.loc[iTrial, 'waitingTime'] = k
    mySession.loc[iTrial, 'isChoiceLeft'] = a

    # R
    mySession.loc[iTrial, 'isCorrect'] = a == (mySession.loc[iTrial, 'stim'] > stim_set.mean())
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[
        iTrial, 'isCatch'] and k > d
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = m*r
    # tau = ITI + r*d * (1-r)*k
    tau = d if mySession.loc[iTrial, 'isRewarded'] else k
    tau = tau + ITI
    mySession.loc[iTrial, 'trialDur'] = tau

    # S'

    # A'
    delta_rho = grad_rho(rho=rhoHat,r=r,tau=tau,m=m)
    delta_b, delta_m, delta_pi, delta_beta = grad_psyc(bHat, mHat, xHat, r, k, betaHat, piHat)
    # delta_beta = grad_beta(beta=betaHat, q=q, k=k, r=r, method=beta_method)

    rhoHat = max(1e-6,rhoHat + (1-(1-learnRateRho)**(tau)) * delta_rho)
    betaHat = max(1e-6,betaHat + learnRateL * delta_beta)
    bHat = bHat + learnRateL * delta_b
    mHat = mHat + learnRateL * delta_m
    piHat = piHat + learnRateL * delta_pi

    mySession.loc[iTrial + 1, 'rho'] = rhoHat
    mySession.loc[iTrial + 1, 'beta'] = betaHat
    mySession.loc[iTrial + 1, 'b'] = bHat
    mySession.loc[iTrial + 1, 'm'] = mHat
    mySession.loc[iTrial + 1, 'pi'] = piHat

    assert(not np.isnan(mySession.loc[iTrial,:].values.astype(float)).any()), "nan found in {}".format(mySession.columns[np.isnan(mySession.loc[iTrial,:].values.astype(float))].values)

    iTrial += 1
    # break

# with open('mod01_cuedPb_nlog10Eta_{:2.1f}.pickle'.format(-np.log10(learnRateRho)),'wb') as fhandle:
#     pickle.dump(mySession,fhandle,-1)

#%% Plotting

hf_learning, ha_learning = plt.subplots(3, 2,figsize=(5,6))

ha_learning[0,0].plot(mySession.rho)
# ha_learning[0,0].plot(np.divide(np.cumsum(mySession.Reward),np.cumsum(mySession.trialDur)))
ha_learning[0,0].plot(np.divide(mySession.Reward.rolling(window=int(nTrials/10)).mean(),mySession.trialDur.rolling(window=int(nTrials/10)).mean()))
ha_learning[0,0].set_xlabel('trial #')
ha_learning[0,0].set_ylabel('rho \n (avg rwd rate)')

ha_learning[0,1].plot(mySession.beta)
ha_learning[0,1].plot(Beta*np.ones(mySession.beta.shape),label='true')#,color='xkcd:gray'
ha_learning[0,1].set_xlabel('trial #')
ha_learning[0,1].set_ylabel('beta \n (mean rwd delay)')

ha_learning[1,0].plot(expit(mySession.pi))
ha_learning[1,0].plot((1-pCatch)*np.ones(mySession.pi.shape),label='true')#,color='xkcd:gray'
ha_learning[1,0].set_xlabel('trial #')
ha_learning[1,0].set_ylabel('pi\n(1-P(catch))')
ha_learning[1,0].set_ylim(-.1,1.1)

for istim, stim in enumerate(stim_set):
    ha_learning[1,1].plot(mySession.waitingTime.loc[mySession.stim==stim])
ha_learning[1,1].set_xlabel('trial #')
ha_learning[1,1].set_ylabel('k \n (waiting time)')

ha_learning[2,0].plot(mySession.b)
ha_learning[2,0].plot(stim_set.mean()*np.ones(mySession.b.shape),label='true')#,color='xkcd:gray'
ha_learning[2,0].set_xlabel('trial #')
ha_learning[2,0].set_ylabel('b \n (decision boundary)')

ha_learning[2,1].plot(mySession.m)
ha_learning[2,1].plot(1.6/stim_noise*np.ones(mySession.m.shape),label='true')#,color='xkcd:gray'
ha_learning[2,1].set_xlabel('trial #')
ha_learning[2,1].set_ylabel('m \n (decision sensitivity)')

hf_learning.show()