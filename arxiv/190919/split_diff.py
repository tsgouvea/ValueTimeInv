# %%
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
sys.version

# %%
np.random.seed(42)

# %%
def makephi(T=600, m=20, h=None, sigma=.2, isNormalized=True, isFlat=False):
    if h == None:
        h = T / 3
    t = np.arange(T)
    if isFlat:
        x = np.full(T, 1. / T)
    else:
        if True:
            i = np.linspace(0, 1, m)
            y = (.5 ** (1 / h)) ** t  # halves every h s
            y_tile = np.tile(y, (len(i), 1))
            i_tile = np.tile(i.reshape(-1, 1), (1, len(y)))
            u = y_tile - i_tile
            x = np.exp(-u ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) * y_tile
        else:
            i = np.linspace(.1, 10, m)
            # i = np.log10(np.logspace(2,100,m))
            y = (1 + t) / T * 2 * np.pi
            y_tile = np.tile(y, (len(i), 1))
            i_tile = np.tile(i.reshape(-1, 1), (1, len(y)))
            u = y_tile * i_tile
            x = np.cos(u)
        if isNormalized:
            x = skp.minmax_scale(x)
            x = x / np.tile(np.sum(x, axis=0).reshape(-1, 1).T, (x.shape[0], 1))
    return x

def truncExp(delayDurMean, delayDurMin, delayDurMax):
    delayDur = delayDurMin - 1
    while (delayDur < delayDurMin) or (delayDur > delayDurMax):
        delayDur = np.random.exponential(delayDurMean)
    return delayDur

def pnorm(z):
    if (z < 0).any():
        z -= z.min()
    z /= z.sum()
    #     z=z**2
    #     z/=z.sum()
    return z.ravel()

# %%
tbf = makephi(m=10, T=20, isNormalized=True)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.imshow(tbf, aspect='auto')
plt.colorbar()
plt.subplot(122)
plt.plot(tbf.T, alpha=.8)
plt.show()
pass

#%% PARAMS
nTrials = 20000
learnRateWait = .01
learnRateRho = .001
pCatch = .1
isCatch = np.random.rand(nTrials) < pCatch
# listStim = np.array([.05, .3, .45, .55, .7, .95])

#%%


#%%
stim_pairs = [[.05,.95],[.30,.70],[.45,.55]]
ucurv = pd.DataFrame(index=np.sort(np.array(stim_pairs).ravel()),columns=['corr','incorr'])

hf = [[]]*len(stim_pairs)
ha = [[]]*len(stim_pairs)

for ipair,stim_pair in enumerate(stim_pairs):
    print(ipair,stim_pair)

    kernel = np.ones((tbf.shape[0], 1))

    m = np.random.choice(stim_pair, nTrials)
    xi = np.random.randn(nTrials) * .18
    # mprime = m + xi
    mprime = np.array(stim_pair)[np.argmin(abs(np.tile(stim_pair,(nTrials,1))-np.tile((m+xi).reshape(-1,1),(1,2))),axis=1)]

    isChoiceLeft = mprime > 0.5
    isChoiceRight = np.logical_not(isChoiceLeft)
    isChoiceCorrect = (m > 0.5) == isChoiceLeft
    isRewarded = np.full(nTrials, False)

    waitingTime = np.full(nTrials, np.nan)
    waitingTime[0] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm(np.dot(kernel.T, tbf))).item()

    feedbackTime = np.full(nTrials, np.nan)
    feedbackTime[0] = truncExp(1.5, .5, 8)

    rho = np.full(nTrials, np.nan)
    rho[0] = 0.1

    delta = np.full(nTrials, np.nan)

    # %%
    hf[ipair], ha[ipair] = plt.subplots(1, 3, figsize=(10, 3))

    # %%
    for iTrial in range(nTrials):
        ## S
        # m = np.random.choice(listStim,nTrials)
        # xi = np.random.randn(nTrials)*.18
        # mprime = m+xi

        ## A
        # isChoiceLeft = mprime>0.5
        # isChoiceRight = np.logical_not(isChoiceLeft)

        ## R
        # isChoiceCorrect = (m>0.5)==isChoiceLeft
        # isCatch = np.random.rand(nTrials) < pCatch

        isRewarded[iTrial] = isChoiceCorrect[iTrial] and not isCatch[iTrial] and waitingTime[iTrial] > feedbackTime[iTrial]

        if iTrial % 3000 == 0:
            print(iTrial)
            # ha[ipair][0].plot(pnorm(np.dot(kernel.T,tbf)))
            ha[ipair][0].plot(pnorm(np.dot(kernel.T, tbf).ravel()))

        ## S'
        feedbackTime[iTrial + 1] = truncExp(1.5, .5, 8)

        ## A'

        tau = feedbackTime[iTrial] if isRewarded[iTrial] else waitingTime[iTrial]
        delta[iTrial] = isRewarded[iTrial].astype(float) - rho[iTrial] * tau
        #     kernel = kernel + learnRateWait*delta[iTrial]*tbf[:,int(waitingTime[iTrial])].reshape((-1,1))
        kernel = kernel + learnRateWait * delta[iTrial] * tbf[:, int(tau)].reshape((-1, 1))
        #     waitingTime[iTrial+1] = np.random.choice(np.arange(tbf.shape[1]),1,p=skp.normalize(skp.minmax_scale(np.dot(kernel.T,tbf),axis=1),axis=1,norm='l1').ravel()).item()
        waitingTime[iTrial + 1] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm(np.dot(kernel.T, tbf))).item()
        rho[iTrial + 1] = rho[iTrial] + (1 - (1 - learnRateRho) ** tau) * delta[iTrial]
        #     break

        if iTrial + 2 == nTrials: break

    ha[ipair][1].cla()
    ha[ipair][1].hist(delta[np.logical_not(np.isnan(delta))], bins=100)
    ha[ipair][1].set_xlim(-2.2, 1.2)
    ha[ipair][2].plot(rho)
    hf[ipair].show()
    for istim, stim in enumerate(stim_pair):
        ucurv.loc[stim, 'corr'] = waitingTime[np.logical_and(isChoiceCorrect, m == stim)].mean()
        ucurv.loc[stim, 'incorr'] = waitingTime[np.logical_and(np.logical_not(isChoiceCorrect), m == stim)].mean()


#%%
#%%
psyc = np.full(np.unique(m).shape, np.nan)
for i, im in enumerate(np.unique(m)):
    psyc[i] = np.mean(mprime[m == im] > .5)

plt.scatter(np.unique(m), psyc)
plt.show()
# plt.plot()

#%%
df = pd.DataFrame({'waitingTime': waitingTime, 'prevCorr': np.hstack((False, isChoiceCorrect[:-1]))})
df.pivot_table(columns='prevCorr')