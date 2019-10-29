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

np.random.seed(42)

sys.version

#%% PARAMS
nTrials = 30000
learnRateWait = .01
learnRateRho = .001
pCatch = 0.1
# listStim = np.array([.05, .3, .45, .55, .7, .95])

#%%
stim_pairs = [[5,95],[30,70],[45,55]]
stims = np.sort(np.array(stim_pairs).ravel())
# ucurv = pd.DataFrame(index=stims,columns=['corr','incorr'])

# hf = [[]]*len(stim_pairs)
# ha = [[]]*len(stim_pairs)


#%%

m = np.random.choice(stims, nTrials)
mprime = stims[np.argmin(abs(np.tile(stims,(nTrials,1))-np.tile((m+np.random.randn(nTrials) * 25).reshape(-1,1),(1,len(stims)))),axis=1)]

mySession = pd.DataFrame({'stim':m,'perc':mprime,'isChoiceLeft': mprime > 50})
mySession.loc[:,'isCorrect'] = (m > 50) == mySession.isChoiceLeft
mySession.loc[:,'isRewarded'] = False
mySession.loc[:,'waitingTime'] = np.nan
mySession.loc[:,'feedbackTime'] = np.nan
mySession.loc[:,'isCatch'] = np.random.rand(nTrials) < pCatch
mySession.loc[:,'abs_stim'] = abs(50 - mySession.loc[:,'stim'])

for istim, stim in enumerate(stims):
    mySession.loc[mySession.stim==stim,'istim'] = istim
    mySession.loc[mySession.perc == stim, 'iperc'] = istim
mySession.loc[:,'istim'] = mySession.loc[:,'istim'].astype(int)
mySession.loc[:, 'iperc'] = mySession.loc[:, 'iperc'].astype(int)

rho = pd.Series(index=np.arange(nTrials))#,columns=np.sort(mySession.istim.drop_duplicates()))
rho.loc[0] = 0

# W = pd.DataFrame(index=np.arange(tbf.shape[0]),columns=rho.columns)
# W.loc[:,:] = 1

K = pd.Series(index=np.sort(mySession.istim.drop_duplicates()))
K.loc[:] = 1

#%%
# sns.regplot(x='stim',y='isChoiceLeft',data=mySession,logistic=True,ci=None,y_jitter=0.01)
# plt.show()

#% INITIAL CONDITIONS

# mySession.loc[0,'waitingTime'] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm((tbf.T @ W).values[:,mySession.iperc[0]])).item()
#
# mySession.loc[0,'feedbackTime'] = truncExp(1.5, .5, 8)

#%
    # hf[ipair], ha[ipair] = plt.subplots(1, 3, figsize=(10, 3))

# %%
#
for iTrial in range(nTrials):
    ## S
    mySession.loc[iTrial, 'feedbackTime'] = truncExp(1.5, .5, 8)

    ## A
    mySession.loc[iTrial, 'waitingTime'] = K.loc[mySession.iperc[iTrial]]
    # waitingTime[iTrial + 1] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm(np.dot(kernel.T, tbf))).item()

    ## R
    mySession.loc[iTrial,'isRewarded'] = mySession.loc[iTrial,'isCorrect'] and not mySession.loc[iTrial,'isCatch'] and mySession.loc[iTrial,'waitingTime'] > mySession.loc[iTrial,'feedbackTime']
    if iTrial % 500 == 0:
        print(iTrial)
        # sns.regplot(x='stim', y='isChoiceLeft', data=mySession, logistic=True, ci=None, ax=ha[0])
        #
        # sns.regplot(x='abs_stim', y='waitingTime', ax=ha[1],
        #             data=mySession.loc[mySession.isCorrect, :],
        #             color='xkcd:leaf green', ci=None, x_bins=stims)
        #
        # sns.regplot(x='abs_stim', y='waitingTime', ax=ha[1],
        #             data=mySession.loc[np.logical_not(mySession.isCorrect), :],
        #             color='xkcd:brick', ci=None, x_bins=stims)


    ## S'


    ## A'
    tau = mySession.loc[iTrial,'feedbackTime'] if mySession.loc[iTrial,'isRewarded'] else mySession.loc[iTrial,'waitingTime']
    delta = mySession.loc[iTrial,'isRewarded'].astype(float) - rho.loc[iTrial] * tau
    expRet = (1-np.exp(mySession.loc[iTrial,'waitingTime']/(-1.5)))*mySession.loc[mySession.loc[:,'perc']==mySession.loc[iTrial,'perc'],'isRewarded'].loc[:iTrial].mean() - rho.loc[iTrial] * tau
    # if iTrial > 5000 and delta < 0: break

    K.loc[mySession.iperc[iTrial]] = K.loc[mySession.iperc[iTrial]] + learnRateWait * delta - expRet# * K.loc[mySession.iperc[iTrial]]

    # W.loc[:, mySession.iperc[iTrial]] = W.loc[:,mySession.iperc[iTrial]] + learnRateWait * delta * tbf[:,int(mySession.loc[iTrial,'waitingTime'])]#.reshape(-1,1)


    # rho.loc[iTrial + 1] = rho.loc[iTrial,:]# + (1 - (1 - learnRateRho) ** tau) * delta[iTrial]
    rho.loc[iTrial + 1] = rho.loc[iTrial] + (1 - (1 - learnRateRho) ** tau) * delta

    #     break

    if iTrial + 1 == nTrials: break

#%%
# for i in range(W.shape[1]):
hf, ha = plt.subplots(2, 2)
sns.regplot(x='stim', y='isChoiceLeft', data=mySession, logistic=True, ci=None, ax=ha[0,0],x_bins=stims)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[mySession.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stims)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,1],
            data=mySession.loc[np.logical_not(mySession.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=stims)
# ha[1,0].plot(stims,np.clip(K,0,8))
ha[1,0].plot(stims,K)
ha[1,0].set_xlabel('stim')
ha[1,0].set_ylabel('waiting time')
ha[1,1].plot(rho)
ha[1,1].set_xlabel('trial #')
ha[1,1].set_ylabel('avg rwd rate')
plt.show()

#%
x_bins = [5,20,45]

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
# #%%
# sns.regplot(x='abs_stim', y='isChoiceLeft', data=mySession.loc[mySession.prevShort,:], logistic=True, ci=None, ax=ha[1,1],color='xkcd:black',y_jitter=.01)
# sns.regplot(x='abs_stim', y='isChoiceLeft', data=mySession.loc[np.logical_not(mySession.prevShort),:], logistic=True, ci=None, ax=ha[1,1],color='xkcd:gray',y_jitter=.01)
#
#
# sns.regplot(x='abs_stim', y='isChoiceLeft', data=mySession.loc[mySession.prevShort,:], logistic=True, ci=None, ax=ha[1,1],color='xkcd:leaf green')
# sns.regplot(x='abs_stim', y='isChoiceLeft', data=mySession.loc[np.logical_not(mySession.prevShort),:], logistic=True, ci=None, ax=ha[1,1],color='xkcd:brick')
#
# ndx = np.logical_and(mySession.prevCorr,mySession.isCorrect)
# sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,0],
#             data=mySession.loc[ndx,:],
#             color='xkcd:leaf green', ci=None, x_bins=stims)
# ndx = np.logical_and(mySession.prevCorr,mySession.isCorrect)
#
# sns.regplot(x='abs_stim', y='waitingTime', ax=ha[0,0],
#             data=mySession.loc[np.logical_not(mySession.prevCorr),:],
#             color='xkcd:brick', ci=None, x_bins=stims)