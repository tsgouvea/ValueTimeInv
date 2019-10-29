import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

from locallib import optimwt, grad_rho, grad_beta

#%% PARAMS
np.random.seed(43)

learnRateRho = .0003
learnRateBeta = .01

rho_method='G3'
# rho_method = 'lak14'
beta_method = 'logprob'
# beta_method = 'prob'

nTrials = int(1e4)

pCatch = 0
Beta = 1.5
ITI = 1
m = 30 #reward magnitude
stim_set = np.linspace(50,95,10)/100#
stim_noise = .2

cols = ['isCorrect','isCatch','isRewarded','Reward','Q','rho',
        'beta','waitingTime','feedbackTime','trialDur']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)

mySession.loc[:, 'Q'] = np.random.choice(stim_set, nTrials) # here to be interpreted as p(Reward)
mySession.loc[:, 'feedbackTime'] = np.random.exponential(Beta,nTrials)
mySession.loc[:,['isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch
mySession.loc[:, 'isCorrect'] = np.random.rand(nTrials) < mySession.loc[:, 'Q']

#% Initialization
mySession.loc[0, 'rho'] = .0001
mySession.loc[0, 'beta'] = .0001

#%% Main Loop

iTrial = 0

while (iTrial+1) < nTrials:
    #%%
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}, rho {:1.2f}, delta_rho {:3.2f}, beta {:1.2f}, delta_beta {:3.2f}".format(iTrial + 1, rhoHat, delta_rho, betaHat, delta_beta))

    # S
    q = mySession.loc[iTrial, 'Q']
    betaHat = mySession.loc[iTrial, 'beta']
    rhoHat = mySession.loc[iTrial, 'rho']
    d = mySession.loc[iTrial, 'feedbackTime']

    # A
    # temperature = 100 if iTrial < nTrials/2 else 0.1
    k = optimwt(beta=betaHat,rho=rhoHat,q=q,m=m,method=rho_method)
    mySession.loc[iTrial, 'waitingTime'] = k

    # R
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
    delta_beta = grad_beta(beta=betaHat, q=q, k=k, r=r, method=beta_method)

    rhoHat = max(1e-6, rhoHat + (1-(1-learnRateRho)**(tau)) * delta_rho)
    betaHat = max(1e-6, betaHat + learnRateBeta * delta_beta)

    mySession.loc[iTrial + 1, 'rho'] = rhoHat
    mySession.loc[iTrial + 1, 'beta'] = betaHat

    assert(not np.isnan(mySession.loc[iTrial,:].values.astype(float)).any()), "nan found in {}".format(mySession.columns[np.isnan(mySession.loc[iTrial,:].values.astype(float))].values)

    iTrial += 1
    # break

# with open('mod01_cuedPb_nlog10Eta_{:2.1f}.pickle'.format(-np.log10(learnRateRho)),'wb') as fhandle:
#     pickle.dump(mySession,fhandle,-1)

#%% Plotting
mySession.loc[:,'early'] = mySession.index < nTrials/2
mySessionLate = mySession.loc[np.logical_not(mySession.early),:].dropna()

hf_summ, ha_summ = plt.subplots(1, 2,figsize=(5.4,2.2))

ha_summ[0].plot(mySession.rho,label='estimated')
ha_summ[0].plot(np.divide(mySession.Reward.rolling(window=int(nTrials/10)).mean(),mySession.trialDur.rolling(window=int(nTrials/10)).mean()),label='true')
ha_summ[0].set_xlabel('trial #')
ha_summ[0].set_ylabel('rho \n (avg rwd rate)')
ha_summ[0].legend(frameon=False)

ha_summ[1].plot(mySession.beta,label='estimated')
ha_summ[1].plot(Beta*np.ones(mySession.beta.shape),label='true')#,color='xkcd:gray'
# ha_summ[1].plot(np.divide(mySession.Reward.rolling(window=int(nTrials/10)).mean(),mySession.trialDur.rolling(window=int(nTrials/10)).mean()),label='observed')
ha_summ[1].set_xlabel('trial #')
ha_summ[1].set_ylabel('beta \n (avg rwd delay)')
ha_summ[1].legend(frameon=False)


# sns.regplot(x='Q', y='waitingTime', data=mySessionLate, logistic=False, ci=None,
#             ax=ha_summ[1],x_bins=stim_set, truncate=True, fit_reg=False, scatter_kws={'label':'simulation'})
# ha_summ[1].set_xlabel('P(B=1)')
# xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
# yaxis = optimwt(beta=mySessionLate.beta.mean(),rho=mySessionLate.rho.mean(),q=xaxis,m=m,method=rho_method)
# yaxis[yaxis<0]=0
# ha_summ[1].plot(xaxis,
#                          yaxis,
#                          color='xkcd:silver',label='decision rule')
# ha_summ[1].legend(fontsize='small',frameon=False)


hf_summ.show()

hf_summ.savefig('fig_cuedPb_learn_rho_beta_{}.png'.format(beta_method))

# #%% asfreq
#
# # mySess2 = mySession.loc[:,['early','rho','Reward','trialDur','Q','waitingTime']].copy()
# mySess2 = mySession.loc[:,['Reward','trialDur','rho','Q','waitingTime','beta','isCorrect']].dropna().copy()
# mySess2.index=pd.to_datetime(pd.to_timedelta(np.cumsum(mySess2.loc[:,'trialDur'].round(decimals=0)),unit='s'))
# # mySess2.index = [t.strftime("%M:%S") for t in mySess2.index]
# mySess3 = mySess2.Reward.dropna().asfreq(freq='S',fill_value=0)
# # mySession.loc[:,'early'] = mySession.index < nTrials/2
# mySessionLate = mySess2.iloc[int(len(mySess2)/2):,:].dropna()
#
# # mySession.loc[:,'early'] = mySession.index < nTrials/2
# # mySessionLate = mySession.loc[np.logical_not(mySession.early),:].dropna()
#
# hf_summ, ha_summ = plt.subplots(1, 2,figsize=(1*np.array([5.4,2.2])))
#
# ha_summ[0].plot(np.divide(mySess2.Reward,mySess2.trialDur).rolling(window=int(nTrials/10)).mean(),label='observed_old')
# ha_summ[0].plot(np.divide(mySess2.Reward.rolling(window=int(nTrials/10)).mean(),mySess2.trialDur.rolling(window=int(nTrials/10)).mean()),label='observed_old2')
# ha_summ[0].plot(mySess3.rolling(window=2400).mean(),label='observed_new')
# ha_summ[0].set_xlabel('trial #')
# ha_summ[0].set_ylabel('rho \n (avg rwd rate)')
# ha_summ[0].plot(mySess2.index,mySess2.rho,label='estimated')
# ha_summ[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=int(mySession.trialDur.sum()/60/4)))
# ha_summ[0].xaxis.set_major_formatter(mdates.DateFormatter("%Hh%M"))
#
# ha_summ[0].legend(frameon=False)
#
# sns.regplot(x='Q', y='waitingTime', data=mySessionLate, logistic=False, ci=None,
#             ax=ha_summ[1],x_bins=stim_set, truncate=True, fit_reg=False, scatter_kws={'label':'simulation'})
# ha_summ[1].set_xlabel('P(B=1)')
# xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
# yaxis = optimwt(beta=mySessionLate.beta.mean(),rho=mySessionLate.rho.mean(),q=xaxis,m=m,method=rho_method)
# yaxis[yaxis<0]=0
# ha_summ[1].plot(xaxis,
#                          yaxis,
#                          color='xkcd:silver',label='decision rule')
# ha_summ[1].legend(fontsize='small',frameon=False)
#
#
# hf_summ.show()
#
# #%%
#
# hf_calibr, ha_calibr = plt.subplots(1, 2,figsize=(1*np.array([5.4,2.2])))
#
# ha_calibr[0].plot(np.divide(mySess2.Reward,mySess2.trialDur).rolling(window=int(nTrials/10)).mean(),label='observed_old')
# ha_calibr[0].plot(np.divide(mySess2.Reward.rolling(window=int(nTrials/10)).mean(),mySess2.trialDur.rolling(window=int(nTrials/10)).mean()),label='observed_old2')
# ha_calibr[0].plot(mySess3.rolling(window=2400).mean(),label='observed_new')
# ha_calibr[0].set_xlabel('trial #')
# ha_calibr[0].set_ylabel('rho \n (avg rwd rate)')
# ha_calibr[0].plot(mySess2.index,mySess2.rho,label='estimated')
# ha_calibr[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=int(mySession.trialDur.sum()/60/4)))
# ha_calibr[0].xaxis.set_major_formatter(mdates.DateFormatter("%Hh%M"))
#
# ha_calibr[0].legend(frameon=False)
#
# temp = mySessionLate.waitingTime
# temp=temp[np.logical_not(np.isinf(temp))]
# x_bins=np.percentile(temp,np.linspace(0,100,30))
# sns.regplot(x='waitingTime', y='isCorrect', data=mySessionLate, logistic=False, ci=None,
#             ax=ha_calibr[1],x_bins=x_bins, truncate=True, fit_reg=False)
# # ha_calibr[1].set_xlabel('P(B=1)')
# # xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
# # yaxis = optimwt(beta=mySessionLate.beta.mean(),rho=mySessionLate.rho.mean(),q=xaxis,m=m,method=rho_method)
# # yaxis[yaxis<0]=0
# # ha_calibr[1].plot(xaxis,
# #                          yaxis,
# #                          color='xkcd:silver',label='decision rule')
# ha_calibr[1].legend(fontsize='small',frameon=False)
#
#
# hf_calibr.show()