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
from locallib import grad_beta, grad_rho, grad_l

#%% PARAMS
np.random.seed(43)

learnRateB = .01
learnRateRho = .01
learnRateBeta = 0

nTrials = int(1e5)

pCatch = 0
Beta = 1
ITI = 1
rewMag = 1
stim_set = np.arange(10)
P_B = pd.Series(index=stim_set,data=np.linspace(5,95,len(stim_set))/100)
L_hat = pd.Series(index=stim_set,dtype=float)

cols = ['isCorrect','isCatch','isRewarded','Reward','P(B)','rho',
        'beta','waitingTime','feedbackTime','trialDur']

mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)

mySession.loc[:,'stim'] = np.random.choice(stim_set,nTrials)
mySession.loc[:, 'P(B)'] = P_B.values[mySession.stim]
mySession.loc[:, 'isCorrect'] = np.random.rand(nTrials) < mySession.loc[:, 'P(B)']
mySession.loc[:, 'feedbackTime'] = np.random.exponential(Beta,nTrials)
mySession.loc[:,['isCatch','isRewarded']] = False
mySession.loc[:, 'isCatch'] = np.random.rand(nTrials) < pCatch


#% Initialization
mySession.loc[0, 'rho'] = .001
mySession.loc[0, 'beta'] = Beta
L_hat.loc[:] = 0 # logit transformed

#% Main Loop

iTrial = 0

while (iTrial+1) < nTrials:
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}, beta {:1.2f}, rho {:1.2f}, L_hat {}".format(iTrial + 1, betaHat, rhoHat, np.round(expit(L_hat.values),2)))

    # S
    x = mySession.loc[iTrial, 'stim']
    l = L_hat.loc[x]
    pRew = expit(l) # logistic transforms back to probability
    betaHat = mySession.loc[iTrial, 'beta']
    rhoHat = mySession.loc[iTrial, 'rho']
    f = mySession.loc[iTrial, 'feedbackTime']

    # A
    # temperature = 100 if iTrial < nTrials/2 else 0.1
    k = 10#max(1e-6, betaHat * np.log(pRew / (betaHat * rhoHat)))

    mySession.loc[iTrial, 'waitingTime'] = k

    # R
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect'] and not mySession.loc[
        iTrial, 'isCatch'] and k > f
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = rewMag*r
    mySession.loc[iTrial, 'trialDur'] = k  # f if mySession.loc[iTrial, 'isRewarded'] else k
    # mySession.loc[iTrial, 'trialDur'] = mySession.loc[iTrial, 'trialDur'] + ITI
    tau = mySession.loc[iTrial, 'trialDur'] + ITI  # + 1e-9

    # S'

    # A'
    delta_rho = grad_rho(rhoHat,r,tau)
    delta_l = grad_l(l,r,k,betaHat)

    rhoHat = rhoHat + learnRateRho * delta_rho
    l = l + learnRateB * delta_l

    rhoHat = max(1e-6,rhoHat)

    mySession.loc[iTrial + 1, 'rho'] = rhoHat
    mySession.loc[iTrial + 1, 'beta'] = betaHat
    L_hat.loc[x] = l

    assert(not np.isnan(mySession.loc[iTrial,:].values.astype(float)).any()), "nan found in {}".format(mySession.columns[np.isnan(mySession.loc[iTrial,:].values.astype(float))].values)
    assert (rhoHat >= 0), "rhoHat < 0 (={:0.2f})".format(rhoHat)

    iTrial += 1
    # break

# with open('mod01_cuedPb_nlog10Eta_{:2.1f}.pickle'.format(-np.log10(learnRateRho)),'wb') as fhandle:
#     pickle.dump(mySession,fhandle,-1)

#%% Plotting

mySession.loc[:,'prevCorr'] = np.hstack((False,mySession.isCorrect.iloc[:-1]))
# mySession.loc[:,'prevShort'] = np.hstack((False,mySession.waitingTime.iloc[:-1]<mySession.waitingTime.median()))
mySession.loc[:,'prevRwd'] = np.hstack((False,mySession.isRewarded.iloc[:-1]))
mySession.loc[1:nTrials, 'prevStim'] = mySession.loc[0:nTrials - 2, 'P(B)'].values
mySession.loc[:,'early'] = mySession.index < nTrials/2
mySessionLate = mySession.loc[np.logical_not(mySession.early),:].dropna()
if False:
    # mySessionLate.loc[:,'early'] = mySessionLate.index < mySessionLate.index.values.mean()
    mySessionLate.loc[:,'long_wt'] = False
    mySessionLate.loc[:, 'long_wt'] = mySessionLate.loc[:, 'waitingTime'] > np.median(mySessionLate.loc[:, 'waitingTime'])
    # for istim, stim in enumerate(stim_set):
    #     ndx = mySessionLate.loc[:,'P(B)'] == stim
    #     mySessionLate.loc[ndx, 'long_wt'] = mySessionLate.loc[ndx, 'waitingTime'] > np.median(mySessionLate.loc[ndx, 'waitingTime'])
else:
    mySessionLate.loc[:, 'early'] = mySessionLate.index < mySessionLate.index.values.mean()
    mySessionLate.loc[:, 'short_wt'] = False
    mySessionLate.loc[:, 'short_wt'] = mySessionLate.loc[:, 'waitingTime'] < np.median(mySessionLate.loc[:, 'waitingTime'])
    # mySessionLate.loc[:, 'short_wt'] = mySessionLate.loc[:, 'waitingTime'] < np.percentile(
    #     mySessionLate.loc[:, 'waitingTime'],75)
    mySessionLate.loc[:, 'prevShort'] = np.hstack((False,mySessionLate.short_wt.iloc[:-1]))
    mySessionLate.loc[:, 'prevWT'] = np.hstack((np.nan, mySessionLate.waitingTime.iloc[:-1]))
    mySessionLate.loc[:, 'long_wt'] = np.logical_not(mySessionLate.short_wt.values)

#%
x_bins = P_B#np.unique(abs((stim_set-stim_set.mean())))

hf_lakS3, ha_lakS3 = plt.subplots(2, 3, figsize=(8.1,5.4),sharex='col',sharey='row')

sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.early,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':'-','label':'Correct (early trials)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.early,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':','label':'Error (early trials)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.early),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':'-','label':'Correct (late trials)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,0],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.early),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':','label':'Error (late trials)'})
ha_lakS3[0,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='P(B)', y='isCorrect', data=mySessionLate.loc[mySessionLate.early,:],truncate=True,
            logistic=False, ci=None, ax=ha_lakS3[1,0],color='xkcd:black',
            x_bins=x_bins,label='Early trials')
sns.regplot(x='P(B)', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.early),:],truncate=True,
            logistic=False, ci=None, ax=ha_lakS3[1,0],color='xkcd:gray',
            x_bins=x_bins,label='Late trials')
ha_lakS3[1,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':'-','label':'Correct (after correct)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':','label':'Error (after correct)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':'-','label':'Correct (after error)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':','label':'Error (after error)'})
ha_lakS3[0,1].legend(frameon=False,fontsize='x-small')

sns.regplot(x='P(B)', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevCorr,:],truncate=True,
            logistic=False, ci=None, ax=ha_lakS3[1,1],color='xkcd:black',
            x_bins=x_bins,label='After correct')
sns.regplot(x='P(B)', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevCorr),:],truncate=True,
            logistic=False, ci=None, ax=ha_lakS3[1,1],color='xkcd:gray',
            x_bins=x_bins,label='After error')
ha_lakS3[1,1].legend(frameon=False,fontsize='x-small')


sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':'-','label':'Correct (after short)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':','label':'Error (after short)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':'-','label':'Correct (after long)'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_lakS3[0,2],truncate=True,
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':','label':'Error (after long)'})
ha_lakS3[0,2].legend(frameon=False,fontsize='x-small')

sns.regplot(x='P(B)', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevShort,:],
            logistic=False, ci=None, ax=ha_lakS3[1,2],color='xkcd:black',truncate=True,
            x_bins=x_bins,label='After short')
sns.regplot(x='P(B)', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevShort),:],
            logistic=False, ci=None, ax=ha_lakS3[1,2],color='xkcd:gray',truncate=True,
            x_bins=x_bins,label='After long')
ha_lakS3[1,2].legend(frameon=False,fontsize='x-small')
hf_lakS3.show()

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
    print("prev_stim:{:1.2f} rho:{:1.2f}".format(stim,mySession.rho[mySession.prevStim == P_B.loc[stim]].mean()))
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
sns.regplot(x='P(B)', y='waitingTime', data=mySessionLate, logistic=False, ci=None,
            ax=ha_waitingTime[0,0],x_bins=P_B, truncate=True)
xaxis = np.linspace(P_B.min(),P_B.max(),100)
yaxis = betaHat * np.log((rewMag*xaxis*(1-pCatch)) / (betaHat * mySessionLate.rho.mean()))
yaxis[yaxis<0]=0
ha_waitingTime[0,0].plot(xaxis,
                         yaxis,
                         color='xkcd:silver',label='agents estimate')
ha_waitingTime[0,0].legend(fontsize='small',frameon=False)
ha_waitingTime[0,1].plot(xaxis,
                         yaxis,
                         color='xkcd:silver',label='agents estimate')
sns.regplot(x='P(B)', y='waitingTime', ax=ha_waitingTime[0,1],truncate=True,
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=P_B, line_kws={'label':'correct'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_waitingTime[0,1],truncate=True,
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=P_B, line_kws={'label':'error'})
# x = mySession.loc[iTrial, 'P(B)']
betaHat = mySession.loc[iTrial, 'beta']
rhoHat = mySession.loc[iTrial, 'rho']
f = mySession.loc[iTrial, 'feedbackTime']
stim_axis = np.linspace(P_B.min(),P_B.max(),91)
bins = np.linspace(mySession.waitingTime.min(),mySession.waitingTime.max(),100)
for i, stim in enumerate(stim_set):
    ha_waitingTime[1,0].plot(mySession.waitingTime[mySession.loc[:,'stim'] == stim])
    ha_waitingTime[1,0].set_xlabel('trial #')
    ha_waitingTime[1,0].set_ylabel('k \n (waitingTime)')
    ha_waitingTime[1,1].hist(mySession.waitingTime[mySession.loc[:,'stim'] == stim],bins=bins,histtype='step')
    ha_waitingTime[1,1].set_xlabel('k \n (waitingTime)')
ha_waitingTime[0,1].legend(frameon=False,fontsize='x-small')
hf_waitingTime.show()

#%
g = sns.lmplot(x="P(B)", y="waitingTime", hue="prevStim", data=mySessionLate,#.loc[mySessionLate.prevRwd, :],
               palette=sns.color_palette('YlOrRd', len(stim_set)), fit_reg=True, logistic=False,
               truncate=True,size=3.5,
               x_bins=P_B,scatter=True, ci=None, scatter_kws={'alpha':.5})
g.fig.show()

g = sns.lmplot(x="P(B)", y="waitingTime", hue="prevStim", data=mySessionLate,#.loc[mySessionLate.prevRwd, :],
               palette=sns.color_palette('YlOrRd', len(stim_set)), fit_reg=True, logistic=False,
               col='prevRwd',truncate=True,size=3.5,
               x_bins=P_B,scatter=True, ci=None, scatter_kws={'alpha':.5})
# g.ax.set_title('After rewarded')
g.fig.show()

hf_summ, ha_summ = plt.subplots(1, 2,figsize=(5.4,2.2))

ha_summ[0].plot(mySession.rho,label='estimated')
ha_summ[0].plot(np.divide(mySession.Reward,mySession.trialDur+ITI).rolling(window=int(nTrials/10)).mean(),label='observed')
ha_summ[0].set_xlabel('trial #')
ha_summ[0].set_ylabel('rho \n (avg rwd rate)')
ha_summ[0].legend(frameon=False)

sns.regplot(x='P(B)', y='waitingTime', data=mySessionLate, logistic=False, ci=None,
            ax=ha_summ[1],x_bins=P_B, truncate=True, fit_reg=False, scatter_kws={'label':'simulation'})
xaxis = np.linspace(P_B.min(),P_B.max(),100)
yaxis = betaHat * np.log((rewMag*xaxis*(1-pCatch)) / (betaHat * mySessionLate.rho.mean()))
yaxis[yaxis<0]=0
ha_summ[1].plot(xaxis,
                         yaxis,
                         color='xkcd:silver',label='decision rule')
ha_summ[1].legend(fontsize='small',frameon=False)


hf_summ.show()
#%
hf_signatures, ha_signatures = plt.subplots(1, 3, figsize=(7.5,2.5))

sns.regplot(x='waitingTime', y='isCorrect', ax=ha_signatures[0], truncate=True,
            data=mySessionLate, logistic=False,
            ci=None, x_bins=np.percentile(mySessionLate.waitingTime,np.linspace(0,100,9)))#np.linspace(mySessionLate.waitingTime.min(),mySessionLate.waitingTime.max(),15))

ha_signatures[0].set_ylim(0.4,1.1)

sns.regplot(x='P(B)', y='waitingTime', ax=ha_signatures[1],truncate=True,
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=P_B, line_kws={'label':'correct'})
sns.regplot(x='P(B)', y='waitingTime', ax=ha_signatures[1],truncate=True,
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=P_B, line_kws={'label':'error'})
ha_signatures[1].legend(frameon=False, fontsize='small')

sns.regplot(x='P(B)', y='isCorrect', ax=ha_signatures[2], truncate=True,
            data=mySessionLate.loc[mySessionLate.long_wt, :], logistic=False,
            color='xkcd:dark blue', ci=None, x_bins=P_B, line_kws={'label': 'long WT'})
sns.regplot(x='P(B)', y='isCorrect', ax=ha_signatures[2], truncate=True,
            data=mySessionLate.loc[np.logical_not(mySessionLate.long_wt), :], logistic=False,
            color='xkcd:light blue', ci=None, x_bins=P_B, line_kws={'label': 'short WT'})
ha_signatures[2].legend(frameon=False, fontsize='small')

hf_signatures.show()
# hf_pp = sns.pairplot(data=mySessionLate.dropna(), vars=['P(B)','waitingTime','prevStim'],hue='prevRwd',diag_kind='kde',palette='YlOrRd')
# hf_pp.fig.show()
