# %%
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
import seaborn as sns

from locallib import makephi, pnorm, truncExp

sys.version

# %%
tbf = makephi(m=10, T=20, isNormalized=True)
plt.figure(figsize=(4, 6))
plt.subplot(211)
plt.imshow(tbf, aspect='auto')
plt.colorbar()
plt.subplot(212)
plt.plot(tbf.T, alpha=.8)
plt.show()

np.random.seed(42)

pass

#%% PARAMS
nTrials = 10000
learnRateWait = .1
learnRateRho = .01
pCatch = 0
# listStim = np.array([.05, .3, .45, .55, .7, .95])

#%%
stim_pairs = [[.05,.95],[.30,.70],[.45,.55]]
stims = np.sort(np.array(stim_pairs).ravel())
# ucurv = pd.DataFrame(index=stims,columns=['corr','incorr'])

# hf = [[]]*len(stim_pairs)
# ha = [[]]*len(stim_pairs)


#%%
m = np.random.choice(stims, nTrials)
mprime = stims[np.argmin(abs(np.tile(stims,(nTrials,1))-np.tile((m+np.random.randn(nTrials) * .25).reshape(-1,1),(1,len(stims)))),axis=1)]

mySession = pd.DataFrame({'stim':m,'perc':mprime,'isChoiceLeft': mprime > 0.5})
mySession.loc[:,'isCorrect'] = (m > 0.5) == mySession.isChoiceLeft
mySession.loc[:,'isRewarded'] = False
mySession.loc[:,'waitingTime'] = np.nan
mySession.loc[:,'feedbackTime'] = np.nan
mySession.loc[:,'isCatch'] = np.random.rand(nTrials) < pCatch
mySession.loc[:,'abs_stim'] = abs(0.5 - mySession.loc[:,'stim'])

for istim, stim in enumerate(stims):
    mySession.loc[mySession.stim==stim,'istim'] = istim
    mySession.loc[mySession.perc == stim, 'iperc'] = istim
mySession.loc[:,'istim'] = mySession.loc[:,'istim'].astype(int)
mySession.loc[:, 'iperc'] = mySession.loc[:, 'iperc'].astype(int)

rho = pd.DataFrame(index=np.arange(nTrials),columns=np.sort(mySession.istim.drop_duplicates()))
rho.loc[0,:] = 1

W = pd.DataFrame(index=np.arange(tbf.shape[0]),columns=rho.columns)
W.loc[:,:] = 1

#%%
sns.regplot(x='stim',y='isChoiceLeft',data=mySession,logistic=True,ci=None,y_jitter=0.01)
plt.show()

#%% INITIAL CONDITIONS

# mySession.loc[0,'waitingTime'] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm((tbf.T @ W).values[:,mySession.iperc[0]])).item()
#
# mySession.loc[0,'feedbackTime'] = truncExp(1.5, .5, 8)

#%%
    # hf[ipair], ha[ipair] = plt.subplots(1, 3, figsize=(10, 3))
hf, ha = plt.subplots(1, 3)

# %%
#
for iTrial in range(nTrials):
    ## S
    mySession.loc[iTrial, 'feedbackTime'] = truncExp(1.5, .5, 8)

    ## A
    mySession.loc[iTrial, 'waitingTime'] = np.random.choice(np.arange(tbf.shape[1]), 1,
                                                            p=pnorm((tbf.T @ W.loc[:,mySession.iperc[iTrial]]))).item()
    # waitingTime[iTrial + 1] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm(np.dot(kernel.T, tbf))).item()

    ## R
    mySession.loc[iTrial,'isRewarded'] = mySession.loc[iTrial,'isCorrect'] and not mySession.loc[iTrial,'isCatch'] and mySession.loc[iTrial,'waitingTime'] > mySession.loc[iTrial,'feedbackTime']
    if iTrial % 1000 == 0:
        print(iTrial)
        sns.regplot(x='stim', y='isChoiceLeft', data=mySession, logistic=True, ci=None, ax=ha[0])

        sns.regplot(x='abs_stim', y='waitingTime', ax=ha[1],
                    data=mySession.loc[mySession.isCorrect, :],
                    color='xkcd:leaf green', ci=None, x_bins=stims)

        sns.regplot(x='abs_stim', y='waitingTime', ax=ha[1],
                    data=mySession.loc[np.logical_not(mySession.isCorrect), :],
                    color='xkcd:brick', ci=None, x_bins=stims)

    #     # ha[ipair][0].plot(pnorm(np.dot(kernel.T,tbf)))
    #     ha[ipair][0].plot(pnorm(np.dot(kernel.T, tbf).ravel()))

    ## S'


    ## A'
    tau = mySession.loc[iTrial,'feedbackTime'] if mySession.loc[iTrial,'isRewarded'] else mySession.loc[iTrial,'waitingTime']
    delta = mySession.loc[iTrial,'isRewarded'].astype(float) - rho.loc[iTrial,mySession.loc[iTrial,'iperc']] * tau
    #     kernel = kernel + learnRateWait*delta[iTrial]*tbf[:,int(mySession.loc[iTrial,'waitingTime'])].reshape((-1,1))
    W.loc[:, mySession.iperc[iTrial]] = W.loc[:,mySession.iperc[iTrial]] + learnRateWait * delta * tbf[:,int(mySession.loc[iTrial,'waitingTime'])]#.reshape(-1,1)
    # kernel = kernel + learnRateWait * delta[iTrial] * tbf[:, int(tau)].reshape((-1, 1))
    # #     waitingTime[iTrial+1] = np.random.choice(np.arange(tbf.shape[1]),1,p=skp.normalize(skp.minmax_scale(np.dot(kernel.T,tbf),axis=1),axis=1,norm='l1').ravel()).item()

    rho.loc[iTrial + 1,:] = rho.loc[iTrial,:]# + (1 - (1 - learnRateRho) ** tau) * delta[iTrial]
    rho.loc[iTrial + 1, mySession.iperc[iTrial]] = rho.loc[iTrial, mySession.iperc[iTrial]] + (1 - (1 - learnRateRho) ** tau) * delta

    #     break

    if iTrial + 1 == nTrials: break

#%%
ha[2].plot(skp.normalize((tbf.T @ W).values.T,norm='l1').T)
plt.show()

#%%
# ha[ipair][1].cla()
# ha[ipair][1].hist(delta[np.logical_not(np.isnan(delta))], bins=100)
# ha[ipair][1].set_xlim(-2.2, 1.2)
# ha[ipair][2].plot(rho)
# hf[ipair].show()
# for istim, stim in enumerate(stim_pair):
#     ucurv.loc[stim, 'corr'] = waitingTime[np.logical_and(isChoiceCorrect, m == stim)].mean()
#     ucurv.loc[stim, 'incorr'] = waitingTime[np.logical_and(np.logical_not(isChoiceCorrect), m == stim)].mean()


#%%
#%%
# psyc = np.full(np.unique(m).shape, np.nan)
# for i, im in enumerate(np.unique(m)):
#     psyc[i] = np.mean(mprime[m == im] > .5)
#
# plt.scatter(np.unique(m), psyc)
# plt.show()
# # plt.plot()
#
# #%%
# df = pd.DataFrame({'waitingTime': waitingTime, 'prevCorr': np.hstack((False, isChoiceCorrect[:-1]))})
# df.pivot_table(columns='prevCorr')