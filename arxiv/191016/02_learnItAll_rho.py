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

learnRateRho = .01
learnRateBoundary = .01
learnRateSlope = .01

nTrials = int(1e4)

pCatch = 0
Beta = 1
ITI = .1
rewMag = 1
stim_set = np.linspace(5,95,10)/100#np.linspace(50,100,6)/100#
b = np.median(stim_set)
stim_set = np.delete(stim_set,np.where(np.isclose(stim_set,b)))
# stim_set = np.array([55, 70, 95])/100
stim_noise = .20

cols = ['isChoiceLeft','isCorrect','isCatch','isRewarded','Reward','stim','abs_stim','percept',
        'P(B)','rho','beta','waitingTime','feedbackTime','trialDur']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)

mySession.loc[:, 'feedbackTime'] = np.random.exponential(Beta,nTrials)
mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials) # here to be interpreted as p(Reward)
mySession.loc[:, 'abs_stim'] = abs(mySession.loc[:, 'stim'] - b)
mySession.loc[:, 'percept'] = stim_set[np.argmin(abs(np.tile(stim_set,(nTrials,1))-np.tile((mySession.loc[:,'stim']+np.random.randn(nTrials) * stim_noise).values.reshape(-1,1),(1,len(stim_set)))),axis=1)]
mySession.loc[:,['isChoiceLeft','isCorrect','isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch

#% Initialization
mySession.loc[0, 'rho'] = .001
mySession.loc[0, 'beta'] = Beta
mySession.loc[0, 'boundary'] = 0.5#np.random.rand()#*100
mySession.loc[0, 'slope'] = 8.23#np.random.rand()*100

#%% Main Loop

iTrial = 0

while (iTrial+1) < nTrials:
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}, b {:1.2f}, m {:1.2f}, rho {:1.2f}".format(iTrial + 1, bHat, mHat, rhoHat))

    # S
    x = mySession.loc[iTrial, 'stim']
    xHat = mySession.loc[iTrial, 'percept']
    bHat = mySession.loc[iTrial, 'boundary']
    mHat = mySession.loc[iTrial, 'slope']
    pRew = expit(mHat * abs(xHat - bHat))
    betaHat = mySession.loc[iTrial, 'beta']
    rhoHat = mySession.loc[iTrial, 'rho']
    f = mySession.loc[iTrial, 'feedbackTime']
    mySession.loc[iTrial, 'P(B)'] = pRew

    # A
    cho = mHat * (xHat - bHat) > 0
    k = max(0, betaHat * np.log(pRew / (betaHat * rhoHat)))
    mySession.loc[iTrial, 'isChoiceLeft'] = cho
    mySession.loc[iTrial, 'isCorrect'] = cho == (x > b)
    mySession.loc[iTrial, 'waitingTime'] = k

    # R
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[
        iTrial, 'isCatch'] and k > f
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = rewMag*r
    mySession.loc[iTrial, 'trialDur'] = k  # f if mySession.loc[iTrial, 'isRewarded'] else k
    # mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'trialDur'] + ITI
    tau = mySession.loc[iTrial, 'trialDur']  # + 1e-9

    # S'

    # A'
    delta_rho = (r * rewMag) / (tau + ITI) - rhoHat
    delta_bHat = mHat * (r - expit(mHat * abs(xHat - bHat))) * np.sign(bHat - xHat)
    delta_mHat = abs(xHat - bHat) * (r - expit(mHat * abs(xHat - bHat)))
    # delta_mHat = abs(xHat - bHat) * (1 - expit(mHat * abs(xHat - bHat))) * (r + (1 - r) * (
    #             (-expit(mHat * abs(xHat - bHat)) * (1 - pCatch)) / (
    #                 pCatch + (1 - pCatch) * (1 - expit(mHat * abs(xHat - bHat))))))

    # rhoHat = rhoHat + (1 - (1 - learnRateRho) ** (tau + ITI)) * delta_rho
    rhoHat = rhoHat + learnRateRho * delta_rho
    rhoHat = max(1e-6,rhoHat)
    # bHat = bHat + learnRateBoundary * delta_bHat
    # mHat = max(0,mHat + learnRateSlope * delta_mHat)

    mySession.loc[iTrial + 1, 'rho'] = rhoHat
    mySession.loc[iTrial + 1, 'beta'] = betaHat
    mySession.loc[iTrial + 1, 'boundary'] = bHat
    mySession.loc[iTrial + 1, 'slope'] = mHat

    assert(not np.isnan(mySession.loc[iTrial,:].values.astype(float)).any()), "nan found in {}".format(mySession.columns[np.isnan(mySession.loc[iTrial,:].values.astype(float))].values)
    assert (rhoHat >= 0), "rhoHat < 0 (={:0.2f})".format(rhoHat)

    iTrial += 1
    # break

# with open('G2sim_eta5e2.pickle','wb') as fhandle:
#     pickle.dump(mySession,fhandle,-1)

#%% Plotting

mySession.loc[:,'prevCorr'] = np.hstack((False,mySession.isCorrect.iloc[:-1]))
mySession.loc[:,'prevRwd'] = np.hstack((False,mySession.isRewarded.iloc[:-1]))
mySession.loc[1:nTrials, 'prevStim'] = mySession.loc[0:nTrials - 2, 'stim'].values
mySessionLate = mySession.loc[mySession.index > nTrials/2,:].dropna()
mySessionLate.loc[:,'prevShort'] = np.hstack((False,mySessionLate.waitingTime.iloc[:-1] < mySessionLate.waitingTime.median()))
mySessionLate.loc[:,'early'] = mySessionLate.index < mySessionLate.index.values.mean()
mySessionLate.loc[:,'long_wt'] = False
for istim, stim in enumerate(stim_set):
    ndx = mySessionLate.stim == stim
    mySessionLate.loc[ndx, 'long_wt'] = mySessionLate.loc[ndx, 'waitingTime'] > np.median(mySessionLate.loc[ndx, 'waitingTime'])
mySessionLate.loc[:, 'prev_long_wt'] = np.hstack((False, mySessionLate.long_wt.iloc[:-1] < mySessionLate.long_wt.median()))
    # break
#%%
hf_learning, ha_learning = plt.subplots(2, 2)

ha_learning[0,0].plot(mySession.isCorrect.rolling(window=1000).mean())
ha_learning[0,0].set_xlabel('trial #')
ha_learning[0,0].set_ylabel('Correct choices \n (running average)')

sns.regplot(x='stim', y='isChoiceLeft', data=mySessionLate, logistic=True, ci=None, ax=ha_learning[0,1],x_bins=stim_set)
xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
mHat = mySessionLate.slope.mean()
bHat = mySessionLate.boundary.mean()
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

#%% FROM 01_cued_Pb

#%
hf_learning, ha_learning = plt.subplots(2, 2)

ha_learning[0,0].plot(mySession.rho)

# ha_learning[0,0].plot(np.divide(np.cumsum(mySession.Reward),np.cumsum(mySession.trialDur+ITI)))
ha_learning[0,0].plot(np.divide(mySession.Reward,mySession.trialDur+ITI).rolling(window=int(nTrials/10)).mean())
ha_learning[0,0].set_xlabel('trial #')
ha_learning[0,0].set_ylabel('rho \n (avg rwd rate)')

ha_learning[0,1].plot(mySession.beta)
ha_learning[0,1].set_xlabel('trial #')
ha_learning[0,1].set_ylabel('beta \n (mean rwd delay)')

bins = np.linspace(mySession.rho.min(),mySession.rho.max(),20)
for i, stim in enumerate(stim_set):
    print("prev_stim:{:1.2f} rho:{:1.2f}".format(stim,mySession.rho[mySession.prevStim == stim].mean()))
    ha_learning[1,0].plot(mySession.rho[mySession.prevStim == stim])
    ha_learning[1,0].set_xlabel('trial #')
    ha_learning[1,0].set_ylabel('rho')
    ha_learning[1,1].hist(mySession.rho[mySession.prevStim == stim],bins=bins,histtype='step')
    ha_learning[1,1].set_xlabel('rho')

#
# ha_learning[1,1].plot(mySession.slope)
# ha_learning[1,1].set_xlabel('trial #')
# ha_learning[1,1].set_ylabel('m \n (psychometric slope)')

hf_learning.show()

#%
hf_waitingTime, ha_waitingTime = plt.subplots(2, 2)
abs_stim_set = np.unique(abs(stim_set-b))
sns.regplot(x='abs_stim', y='waitingTime', data=mySessionLate, logistic=False, ci=None,
            ax=ha_waitingTime[0,0],x_bins=abs_stim_set, truncate=True)
xaxis = np.linspace(abs_stim_set.min(),abs_stim_set.max(),100)
yaxis = betaHat * np.log(expit(mHat*xaxis) / (betaHat * mySessionLate.rho.mean()))
yaxis[yaxis<0]=0
ha_waitingTime[0,0].plot(xaxis,
                         yaxis,
                         color='xkcd:silver',label='agents estimate')
ha_waitingTime[0,0].legend(fontsize='small',frameon=False)
# ha_waitingTime[0,1].plot(xaxis,
#                          yaxis,
#                          color='xkcd:silver',label='agents estimate')
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_waitingTime[0,1],truncate=True,
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=abs_stim_set, line_kws={'label':'correct'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_waitingTime[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=abs_stim_set, line_kws={'label':'error'})
x = mySession.loc[iTrial, 'abs_stim']
betaHat = mySession.loc[iTrial, 'beta']
rhoHat = mySession.loc[iTrial, 'rho']
f = mySession.loc[iTrial, 'feedbackTime']
stim_axis = np.linspace(abs_stim_set.min(),abs_stim_set.max(),91)
bins = np.linspace(mySession.waitingTime.min(),mySession.waitingTime.max(),100)
for i, stim in enumerate(abs_stim_set):
    ha_waitingTime[1,0].plot(mySession.waitingTime[mySession.loc[:,'abs_stim'] == stim])
    ha_waitingTime[1,0].set_xlabel('trial #')
    ha_waitingTime[1,0].set_ylabel('k \n (waitingTime)')
    ha_waitingTime[1,1].hist(mySession.waitingTime[mySession.loc[:,'abs_stim'] == stim],bins=bins,histtype='step')
    ha_waitingTime[1,1].set_xlabel('k \n (waitingTime)')
ha_waitingTime[0,1].legend(frameon=False,fontsize='x-small')
hf_waitingTime.show()

#%
x_bins = abs_stim_set#np.unique(abs((stim_set-stim_set.mean())))

hf_lakS3, ha_lakS3 = plt.subplots(2, 3, figsize=(8.1,5.4),sharex='col',sharey='row')

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.early,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':'-','label':'Correct (early trials)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.early,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':','label':'Error (early trials)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.early),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':'-','label':'Correct (late trials)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.early),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':','label':'Error (late trials)'})
ha_lakS3[0,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.early,:],truncate=True,
            logistic=True, ci=None, ax=ha_lakS3[1,0],color='xkcd:black',
            x_bins=x_bins,label='Early trials')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.early),:],truncate=True,
            logistic=True, ci=None, ax=ha_lakS3[1,0],color='xkcd:gray',
            x_bins=x_bins,label='Late trials')
ha_lakS3[1,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':'-','label':'Correct (after correct)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':','label':'Error (after correct)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':'-','label':'Correct (after error)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':','label':'Error (after error)'})
ha_lakS3[0,1].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevCorr,:],truncate=True,
            logistic=True, ci=None, ax=ha_lakS3[1,1],color='xkcd:black',
            x_bins=x_bins,label='After correct')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevCorr),:],truncate=True,
            logistic=True, ci=None, ax=ha_lakS3[1,1],color='xkcd:gray',
            x_bins=x_bins,label='After error')
ha_lakS3[1,1].legend(frameon=False,fontsize='x-small')


sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':'-','label':'Correct (after short)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':','label':'Error (after short)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':'-','label':'Correct (after long)'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':','label':'Error (after long)'})
ha_lakS3[0,2].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevShort,:],
            logistic=True, ci=None, ax=ha_lakS3[1,2],color='xkcd:black',truncate=True,
            x_bins=x_bins,label='After short')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevShort),:],
            logistic=True, ci=None, ax=ha_lakS3[1,2],color='xkcd:gray',truncate=True,
            x_bins=x_bins,label='After long')
ha_lakS3[1,2].legend(frameon=False,fontsize='x-small')
hf_lakS3.show()

#%
g = sns.lmplot(x="abs_stim", y="waitingTime", hue="prevStim", data=mySessionLate,#.loc[mySessionLate.prevRwd, :],
               palette=sns.color_palette('YlOrRd', len(stim_set)), fit_reg=True, logistic=False,
               truncate=True,size=3.5,
               x_bins=stim_set,scatter=True, ci=None, scatter_kws={'alpha':.5})
g.fig.show()

g = sns.lmplot(x="abs_stim", y="waitingTime", hue="prevStim", data=mySessionLate,#.loc[mySessionLate.prevRwd, :],
               palette=sns.color_palette('YlOrRd', len(stim_set)), fit_reg=True, logistic=False,
               col='prevRwd',truncate=True,size=3.5,
               x_bins=stim_set,scatter=True, ci=None, scatter_kws={'alpha':.5})
# g.ax.set_title('After rewarded')
g.fig.show()

g = sns.lmplot(x="stim", y="isChoiceLeft", hue="prevStim", data=mySessionLate,#.loc[mySessionLate.prevRwd, :],
               palette=sns.color_palette('YlOrRd', len(stim_set)), fit_reg=True, logistic=True,
               col='prevRwd',truncate=True,size=3.5,
               x_bins=stim_set,scatter=True, ci=None, scatter_kws={'alpha':.5})
# g.ax.set_title('After rewarded')
g.fig.show()

g = sns.lmplot(x="stim", y="isChoiceLeft", hue="prevStim", data=mySessionLate,#.loc[mySessionLate.prevRwd, :],
               palette=sns.color_palette('YlOrRd', len(stim_set)), fit_reg=True, logistic=True,
               col='prevRwd',truncate=True,size=3.5,
               x_bins=stim_set,scatter=True, ci=None, scatter_kws={'alpha':.5})
# g.ax.set_title('After rewarded')
g.fig.show()

hf_summ, ha_summ = plt.subplots(1, 2,figsize=(5.4,2.2))

ha_summ[0].plot(mySession.rho,label='estimated')
ha_summ[0].plot(np.divide(mySession.Reward,mySession.trialDur+ITI).rolling(window=int(nTrials/10)).mean(),label='observed')
ha_summ[0].set_xlabel('trial #')
ha_summ[0].set_ylabel('rho \n (avg rwd rate)')
ha_summ[0].legend(frameon=False)

sns.regplot(x='abs_stim', y='waitingTime', data=mySessionLate, logistic=False, ci=None,
            ax=ha_summ[1],x_bins=stim_set, truncate=True, fit_reg=False, scatter_kws={'label':'simulation'})
# xaxis = np.linspace(abs_stim_set.min(),abs_stim_set.max(),100)
# yaxis = betaHat * np.log(mHat * xaxis / (betaHat * mySessionLate.rho.mean()))
# yaxis[yaxis<0]=0
# ha_summ[1].plot(xaxis,yaxis,color='xkcd:silver',label='decision rule')
ha_summ[1].legend(fontsize='small',frameon=False)


hf_summ.show()

hf_signatures, ha_signatures = plt.subplots(1, 3, figsize=(7.5,2.5))

sns.regplot(x='waitingTime', y='isCorrect', ax=ha_signatures[0], truncate=True,
            data=mySessionLate, logistic=True,
            ci=None, x_bins=np.percentile(mySessionLate.waitingTime,np.linspace(0,100,9)))#np.linspace(mySessionLate.waitingTime.min(),mySessionLate.waitingTime.max(),15))
# ha_signatures[0].legend(frameon=False, fontsize='small')
ha_signatures[0].set_ylim(0.5,1)

sns.regplot(x='abs_stim', y='waitingTime', ax=ha_signatures[1],truncate=True,
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=abs_stim_set, line_kws={'label':'correct'})
sns.regplot(x='abs_stim', y='waitingTime', ax=ha_signatures[1],truncate=True,
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=abs_stim_set, line_kws={'label':'error'})
ha_signatures[1].legend(frameon=False, fontsize='small')

sns.regplot(x='abs_stim', y='isCorrect', ax=ha_signatures[2], truncate=True,
            data=mySessionLate.loc[mySessionLate.long_wt, :], logistic=True,
            color='xkcd:dark blue', ci=None, x_bins=abs_stim_set, line_kws={'label': 'long WT'})
sns.regplot(x='abs_stim', y='isCorrect', ax=ha_signatures[2], truncate=True,
            data=mySessionLate.loc[np.logical_not(mySessionLate.long_wt), :], logistic=True,
            color='xkcd:light blue', ci=None, x_bins=abs_stim_set, line_kws={'label': 'short WT'})
ha_signatures[2].legend(frameon=False, fontsize='small')

hf_signatures.show()