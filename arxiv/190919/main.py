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

# %%
# kernel = 1 + .1*np.random.randn(tbf.shape[0],1)
kernel = np.ones((tbf.shape[0], 1))

# z = skp.normalize(np.dot(kernel.T,tbf),axis=1,norm='l1')
plt.plot(np.dot(kernel.T, tbf).T)
# plt.plot(pnorm(np.dot(kernel.T,tbf).ravel()))
plt.show()

pass

# %% md
plt.hist(np.random.choice(np.arange(20), 10000, p=pnorm(np.dot(kernel.T, tbf))), bins=10)
plt.show()

pass

# %%

nTrials = 10000
learnRateWait = .3
learnRateRho = .01

pCatch = .1
isCatch = np.random.rand(nTrials) < pCatch

listStim = np.array([.05, .3, .45, .55, .7, .95])
m = np.random.choice(listStim, nTrials)
xi = np.random.randn(nTrials) * .18
mprime = m + xi

isChoiceLeft = mprime > 0.5
isChoiceRight = np.logical_not(isChoiceLeft)
isChoiceCorrect = (m > 0.5) == isChoiceLeft
isRewarded = np.full(nTrials, False)

waitingTime = np.full(nTrials, np.nan)
# waitingTime[0] = np.random.choice(np.arange(tbf.shape[1]),1,p=skp.normalize(skp.minmax_scale(np.dot(kernel.T,tbf),axis=1),axis=1,norm='l1').ravel()).item()
waitingTime[0] = np.random.choice(np.arange(tbf.shape[1]), 1, p=pnorm(np.dot(kernel.T, tbf))).item()

feedbackTime = np.full(nTrials, np.nan)
feedbackTime[0] = truncExp(1.5, .5, 8)

rho = np.full(nTrials, np.nan)
rho[0] = 0

delta = np.full(nTrials, np.nan)

# %%
hf, ha = plt.subplots(1, 3, figsize=(10, 3))
plt.show()

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

    if iTrial % 5000 == 0:
        print(iTrial)
        # ha[0].plot(pnorm(np.dot(kernel.T,tbf)))
        ha[0].plot(pnorm(np.dot(kernel.T, tbf).ravel()))

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
#     if iTrial > 10000 and not isRewarded[iTrial]: break
# isChoiceRight = np.logical_not(isChoiceLeft)


# %%

ha[1].cla()
ha[1].hist(delta[np.logical_not(np.isnan(delta))], bins=100)
ha[1].set_xlim(-2.2, 1.2)
ha[2].plot(rho)
hf.show()

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