import numpy as np
import pandas as pd
from sklearn import preprocessing as skp
from scipy.special import expit, lambertw

def makephi(nTimeBins=600, m=20, h=None, sigma=.2, isNormalized=True, isFlat=False, maxT=None):
    columns = np.arange(nTimeBins) if maxT is None else np.linspace(0,maxT,nTimeBins)
    if h == None:
        h = nTimeBins / 3
    t = np.arange(nTimeBins)
    if isFlat:
        x = np.full(nTimeBins, 1. / nTimeBins)
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
            y = (1 + t) / nTimeBins * 2 * np.pi
            y_tile = np.tile(y, (len(i), 1))
            i_tile = np.tile(i.reshape(-1, 1), (1, len(y)))
            u = y_tile * i_tile
            x = np.cos(u)
        if isNormalized:
            x = skp.minmax_scale(x)
            x = x / np.tile(np.sum(x, axis=0).reshape(-1, 1).T, (x.shape[0], 1))
    Phi = pd.DataFrame(index=np.arange(m),columns=columns)
    Phi.loc[:,:] = np.flipud(x)
    return Phi

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

def grad_rho(rho,r,tau,m=1):
    delta_rho = (r * m) / tau - rho
    return delta_rho

def grad_beta(beta,q,k,r,method='logprob'):
    if method=='logprob':
        delta_beta = k * np.exp(-k / beta) / beta**2 * (
                (1 - r) * q / (1 - q*(1- np.exp(-k / beta))) - r / (1 - np.exp(-k / beta)))
    elif method=='prob':
        delta_beta = k * np.exp(-k / beta) / beta ** 2 * q * (-2*r+1)
    else:
        raise RuntimeError('method=\'{}\' not recognized. Try method=\'logprob\''.format(method))
    return delta_beta
#
def grad_psyc(b, m, x, r, k, beta, pi):
    z = m * abs(x - b)
    delta_b = expit(-z) * m * np.sign(b-x)
    delta_m = expit(-z) * np.abs(x - b)
    delta_pi = expit(-pi)
    delta_beta = (-k * np.exp(-k / beta)) / (beta ** 2 * (1 - np.exp(-k / beta)))
    if r==0:
        alpha = expit(z) * expit(pi) * (1 - np.exp(-k / beta))
        alpha_dLdalpha = -1 * alpha / (1 - alpha)
        delta_b *= alpha_dLdalpha
        delta_m *= alpha_dLdalpha
        delta_pi *= alpha_dLdalpha
        delta_beta *= alpha_dLdalpha

    return delta_b, delta_m, delta_pi, delta_beta

    # z = m * abs(x - b)
    # alpha = expit(z) * expit(pi) * (1-np.exp(-k/beta))
    # alpha_dLdalpha = r+(1-r)*alpha/(1-alpha)
    # dLdz = (1-expit(z))*alpha_dLdalpha
    # delta_b = dLdz*m*np.sign(x-b)
    # delta_m = dLdz*np.abs(x-b)
    # delta_pi = (1-expit(pi)) * alpha_dLdalpha
    # delta_beta = (-k*np.exp(-k/beta))/(beta**2*(1-np.exp(-k/beta)))*alpha_dLdalpha
    # return delta_b, delta_m, delta_pi, delta_beta
# #TODO: check Katharina's future-trial effect
#
# def grad_l(l,r,k,beta):
#     # delta_l = (1 - np.exp(-k / beta)) * expit(l) * (1 - expit(l)) * (2 * r - 1)
#     sig = expit(l)
#     exF = 1-np.exp(-k/beta)
#     delta_l = (1-sig) * (r - (1-r)*exF*sig/(1-exF*sig))
#     return delta_l
#
# def grad_w(w,phi,r,k,beta):
#     # delta_l = (1 - np.exp(-k / beta)) * expit(l) * (1 - expit(l)) * (2 * r - 1)
#     sig = expit(w@phi)
#     exF = 1-np.exp(-k/beta)
#     delta_w = (1-sig) * (r - (1-r)*exF*sig/(1-exF*sig)) * phi
#     return delta_w
#
# def grad_phi(w,phi,r,k,beta):
#     # delta_l = (1 - np.exp(-k / beta)) * expit(l) * (1 - expit(l)) * (2 * r - 1)
#     sig = expit(w@phi)
#     exF = 1-np.exp(-k/beta)
#     delta_phi = (1-sig) * (r - (1-r)*exF*sig/(1-exF*sig)) * w
#     return delta_phi

def optimwt(beta=None,rho=None,q=None,m=1,tau=None,method='G3'):
    # rho = .001 * rho
    if method=='G1':
        k = np.real(-beta * lambertw(-np.exp(-(beta + tau) / beta), k=-1) - beta - tau)
    elif method=='G2':
        k = beta * np.log(q / (rho * beta))
    elif method=='G3':
        k = beta * np.log(q * (m - rho * beta) / ((1 - q) * rho * beta))
    elif method=='lak14':
        k = beta * np.log(q * (1 - rho * beta) / (rho * beta * (1 - q)))
    else:
        raise RuntimeError('method=\'{}\' not recognized. Try method=\'G3\''.format(method))
    if np.isscalar(k):
        k = k if k>0 else 1e-12
    else:
        k[k<=0]=1e-12
    return k