import numpy as np
from sklearn import preprocessing as skp

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

def truncExp(delayDurMean=1.5, delayDurMin=0.5, delayDurMax=8, N=1):
    delayDur = np.full(N,float(delayDurMin) - 1)
    while (delayDur < delayDurMin).any() or (delayDur > delayDurMax).any():
        ndx = np.logical_or(delayDur < delayDurMin,delayDur > delayDurMax)
        delayDur[ndx] = np.random.exponential(delayDurMean,ndx.sum())
    delayDur = delayDur.item() if len(delayDur) == 1 else delayDur
    return delayDur

def pnorm(z):
    if (z < 0).any():
        z -= z.min()
    z /= z.sum()
    #     z=z**2
    #     z/=z.sum()
    return z.ravel()