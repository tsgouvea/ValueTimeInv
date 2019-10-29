import pandas as pd
import numpy as np
import scipy.stats as spt
import matplotlib.pyplot as plt
import seaborn as sns

#%% PARAMS
np.random.seed(43)

myMod = True

nTrials = 5000

rewMag = 1
stim_set = np.linspace(-50,50,21)# np.array([5, 30, 45, 55, 70, 95])/100
stim_noise = 20
Phi = pd.DataFrame(index=stim_set,columns=stim_set,dtype=float)
Ql = pd.Series(index=stim_set,dtype=float)
Qr = pd.Series(index=stim_set,dtype=float)
cols = ['isChoiceLeft','isCorrect','isRewarded','Reward','pReward', 'expectedReward','stim','percept',
        'delta_m','delta_fb','Ql','Qr']
mySession = pd.DataFrame(index=np.arange(nTrials), columns=cols, dtype=float)
mySession.loc[:, 'stim'] = np.random.choice(stim_set, nTrials)
mySession.loc[:, 'abs_stim'] = abs(mySession.loc[:, 'stim']-stim_set.mean())
mySession.loc[:,['isChoiceLeft','isCorrect','isRewarded']] = False

#%% Initialization
b = stim_set.mean()

if myMod:
    Phi.loc[:] = np.eye(len(stim_set))
else: # beliefState
    for i in Phi.columns: Phi.loc[:,i] = spt.norm.pdf(Phi.index,i,stim_noise)
    for i in Phi.columns: Phi.loc[:,i] = Phi.loc[:,i]/Phi.loc[:,i].sum()

learnRate = 1/(100*(Phi @ Phi.T).values.ravel().mean()) # apud eq. (9.19) in Sutton & Barto, 2nd ed

Ql.loc[:] = .5
Qr.loc[:] = .5

#%% Main Loop

assert(np.isclose(Phi.sum(axis=0),1).all() and (Phi.values.ravel()>=0).all()),\
    "Columns of Phi must be probability distributions over stimuli"
iTrial = 0
while (iTrial+1) < nTrials:
    if (iTrial + 1) % int(nTrials/100) == 0:
        print("iTrial:{:5.0f}".format(iTrial + 1))

    # S
    x = mySession.loc[iTrial, 'stim']
    p = spt.norm.pdf(Phi.index,x,stim_noise)
    xHat = np.random.choice(stim_set, 1, replace=True, p=p / p.sum()).item()
    phi = Phi.loc[:,xHat].values.ravel()
    mySession.loc[iTrial, 'percept'] = xHat

    # A
    ql = Ql @ phi
    qr = Qr @ phi
    c = ql > qr# if iTrial > nTrials/2 else np.random.rand()>0.5
    # if myMod:
    eRew = ql if c else qr
    # else:
    #     eRew = phi @ ((Phi.T @ Ql > Phi.T @ Qr) == (x > b)).astype(float) * rewMag
    #     # np.diag(Phi.T @ (np.tile((Phi.T @ Ql > Phi.T @ Qr),(len(stim_set),1)).T == (np.tile(stim_set,(len(stim_set),1)) > b)))

    pRew = eRew/rewMag
    mySession.loc[iTrial, 'isChoiceLeft'] = c
    mySession.loc[iTrial, 'isCorrect'] = c == (x > b)
    # R
    mySession.loc[iTrial, 'isRewarded'] = mySession.loc[iTrial, 'isCorrect']
    r = float(mySession.loc[iTrial, 'isRewarded'])
    mySession.loc[iTrial, 'Reward'] = rewMag*r

    # S'

    # A'
    delta_m = ql if c else qr
    delta_m = delta_m - pd.concat((Ql,Qr)).mean()
    delta_fb = r - eRew
    if c:
        Ql.loc[:] = Ql.loc[:] + learnRate * delta_fb * phi
    else:
        Qr.loc[:] = Qr.loc[:] + learnRate * delta_fb * phi
    mySession.loc[iTrial, 'expectedReward'] = eRew
    mySession.loc[iTrial, 'pReward'] = pRew
    mySession.loc[iTrial, 'delta_m'] = delta_m
    mySession.loc[iTrial, 'delta_fb'] = delta_fb
    mySession.loc[iTrial, 'Ql'] = ql
    mySession.loc[iTrial, 'Qr'] = qr
    iTrial += 1

#%% Plotting
# hf_learnWT.show()

mySession.loc[:,'prevCorr'] = np.hstack((False,mySession.isCorrect.iloc[:-1]))
mySession.loc[:,'prevShort'] = np.hstack((False,mySession.delta_m.iloc[:-1]<mySession.delta_m.median()))
mySession.loc[:,'prevRwd'] = np.hstack((False,mySession.isRewarded.iloc[:-1]))
mySession.loc[1:nTrials, 'prevStim'] = mySession.loc[0:nTrials - 2, 'stim']
# mySession.loc[1:nTrials, 'prevStim'] = np.percentile(stim_set,np.linspace(0,100,8))[np.digitize(mySession.loc[0:nTrials - 2, 'stim'].values,np.percentile(stim_set,np.linspace(0,100,8)))-1]
mySessionLate = mySession.loc[mySession.index > nTrials/2,:].dropna()

# hf_learning, ha_learning = plt.subplots(2, 2)
#
# ha_learning[0,0].plot(mySession.rho)
# # ha_learning[0,0].plot(np.divide(np.cumsum(mySession.Reward),np.cumsum(mySession.trialDur+ITI)))
# ha_learning[0,0].set_xlabel('trial #')
# ha_learning[0,0].set_ylabel('rho \n (avg rwd rate)')
#
# ha_learning[0,1].plot(mySession.beta)
# ha_learning[0,1].set_xlabel('trial #')
# ha_learning[0,1].set_ylabel('beta \n (mean rwd delay)')
#
# ha_learning[1,0].plot(mySession.boundary)
# ha_learning[1,0].set_ylim(0,1)
# ha_learning[1,0].set_xlabel('trial #')
# ha_learning[1,0].set_ylabel('bHat \n (decision boundary)')
#
# ha_learning[1,1].plot(mySession.slope)
# ha_learning[1,1].set_xlabel('trial #')
# ha_learning[1,1].set_ylabel('m \n (psychometric slope)')
# hf_learning.show()

#%
hf_DPE, ha_DPE = plt.subplots(2, 2)
sns.regplot(x='stim', y='isChoiceLeft', data=mySessionLate, logistic=True, ci=None, ax=ha_DPE[0,0],x_bins=stim_set)
xaxis = np.linspace(stim_set.min(),stim_set.max(),100)
# ha_DPE[0,0].plot(xaxis,expit(mHat*(xaxis-bHat)),color='xkcd:silver',label='agents estimate')
# ha_DPE[0,0].legend(fontsize='small',frameon=False)
sns.regplot(x='abs_stim', y='delta_m', ax=ha_DPE[0,1],
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stim_set)
sns.regplot(x='abs_stim', y='delta_m', ax=ha_DPE[0,1],
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=stim_set)
sns.regplot(x='abs_stim', y='delta_fb', ax=ha_DPE[1,0],
            data=mySessionLate.loc[mySessionLate.isCorrect, :],
            color='xkcd:leaf green', ci=None, x_bins=stim_set)
sns.regplot(x='abs_stim', y='delta_fb', ax=ha_DPE[1,0],
            data=mySessionLate.loc[np.logical_not(mySessionLate.isCorrect), :],
            color='xkcd:brick', ci=None, x_bins=stim_set)
x = mySession.loc[iTrial, 'stim']
xHat = mySession.loc[iTrial, 'percept']
# bHat = mySession.loc[iTrial, 'boundary']
# mHat = mySession.loc[iTrial, 'slope']
# betaHat = mySession.loc[iTrial, 'beta']
# rhoHat = mySession.loc[iTrial, 'rho']
# f = mySession.loc[iTrial, 'feedbackTime']
x_bins = np.unique(abs((stim_set-stim_set.mean())))
stim_axis = np.linspace(stim_set.min(),stim_set.max(),91)
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.delta_m<mySessionLate.delta_m.median(),:],
            logistic=True, ci=None, ax=ha_DPE[1,1],color='xkcd:red',
            x_bins=x_bins,label='After correct')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.delta_m>mySessionLate.delta_m.median(),:],
            logistic=True, ci=None, ax=ha_DPE[1,1],color='xkcd:light blue',
            x_bins=x_bins,label='After correct')
# ha_DPE[1,0].imshow(Theta.T @ Phi,aspect='auto',cmap='Reds',
#                extent=list(np.hstack((Phi.columns[[0,-1]].values,Theta.columns[[-1,0]].values))))
# ha_DPE[1,0].set_xlabel('waiting time (s)')
# ha_DPE[1,0].set_ylabel('xHat \n (perceived stimulus)')
# ha_DPE[1,0].set_title('Q(delta_m)')
# ha_DPE[1,1].imshow(softmax((Theta.T @ Phi).values/temperature,axis = 1),aspect='auto',cmap='Reds',
#                extent=list(np.hstack((Phi.columns[[0,-1]].values,Theta.columns[[-1,0]].values))))
# ha_DPE[1,1].set_xlabel('waiting time (s)')
# ha_DPE[1,1].set_ylabel('xHat \n (perceived stimulus)')
# ha_DPE[1,1].set_title('P(delta_m)')
hf_DPE.show()

#%


hf_lakS3, ha_lakS3 = plt.subplots(2, 2)

sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, label='Correct (after correct)')
sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevCorr,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':'}, label='Error (after correct)')
sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, label='Correct (after error)')
sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,0],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevCorr),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':'}, label='Error (after error)')
ha_lakS3[0,0].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevCorr,:],
            logistic=True, ci=None, ax=ha_lakS3[1,0],color='xkcd:black',
            x_bins=x_bins,label='After correct')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevCorr),:],
            logistic=True, ci=None, ax=ha_lakS3[1,0],color='xkcd:gray',
            x_bins=x_bins,label='After error')
ha_lakS3[1,0].legend(frameon=False,fontsize='x-small')


sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:black', ci=None, label='Correct (after short)')
sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(mySessionLate.prevShort,np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:black', ci=None, line_kws={'ls':':'}, label='Error (after short)')
sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),mySessionLate.isCorrect),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, label='Correct (after long)')
sns.regplot(x='abs_stim', y='delta_m', ax=ha_lakS3[0,1],
            data=mySessionLate.loc[np.logical_and(np.logical_not(mySessionLate.prevShort),np.logical_not(mySessionLate.isCorrect)),:],
            x_bins=x_bins,color='xkcd:gray', ci=None, line_kws={'ls':':'}, label='Error (after long)')
ha_lakS3[0,1].legend(frameon=False,fontsize='x-small')

sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[mySessionLate.prevShort,:],
            logistic=True, ci=None, ax=ha_lakS3[1,1],color='xkcd:black',
            x_bins=x_bins,label='After short')
sns.regplot(x='abs_stim', y='isCorrect', data=mySessionLate.loc[np.logical_not(mySessionLate.prevShort),:],
            logistic=True, ci=None, ax=ha_lakS3[1,1],color='xkcd:gray',
            x_bins=x_bins,label='After long')
ha_lakS3[1,1].legend(frameon=False,fontsize='x-small')
hf_lakS3.show()

#%
g = sns.lmplot(x="stim", y="isChoiceLeft", hue="prevStim", data=mySessionLate.loc[mySessionLate.prevRwd, :], y_jitter=0.01,
               palette=sns.color_palette('Spectral', len(stim_set)), fit_reg=True, logistic=True, scatter=False, ci=None)
g.fig.show()


# ha[2].plot(skp.normalize((Phi.T @ Theta).values.T,norm='l1').T)

# plt.figure(figsize=(4, 6))
# plt.subplot(211)
# plt.imshow(Phi, aspect='auto')
# # plt.colorbar()
# plt.subplot(212)

# plt.plot(Phi.T, alpha=.8)

hf_ws, ha_ws = plt.subplots(1,1)
ha_ws.plot(Ql)
ha_ws.plot(Qr)
hf_ws.show()
