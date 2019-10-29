import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skp
import seaborn as sns
import tensorflow.keras as kr

from locallib import makephi, pnorm, truncExp

sys.version

# #%% CRITIC
#
# if 'critic' not in locals():
#     critic = kr.models.Sequential()
#     critic.add(kr.layers.Dense(units=10,activation='softplus',input_dim=3))
#     critic.add(kr.layers.Dense(units=10,activation='softplus'))
#     critic.add(kr.layers.Dense(units=1, activation='linear'))
# critic.compile(optimizer='SGD',
#               loss='mse')
# critic.summary()

#%% TOY DATA

np.random.seed(42)

nTrials = 100000
pCatch = 0.1
stim_pairs = [[5,95],[30,70],[45,55]]
stims = np.sort(np.array(stim_pairs).ravel())
stim_noise = 15

m = np.random.choice(stims, nTrials)
mprime = m+np.random.randn(nTrials) * stim_noise

mySession = pd.DataFrame({'stim':m,'perc':mprime,'isChoiceLeft': mprime > 50})
mySession.loc[:,'isCorrect'] = (m > 50) == mySession.isChoiceLeft
mySession.loc[:,'waitingTime'] = abs((m+np.random.randn(nTrials) * stim_noise) - 50)/8#truncExp(1.5,0.5,8,nTrials)
mySession.loc[:,'feedbackTime'] = truncExp(1.5,0.5,8,nTrials)
mySession.loc[:,'isCatch'] = np.random.rand(nTrials) < pCatch
mySession.loc[:,'isRewarded'] = np.logical_and(mySession.loc[:,'isCorrect'],
                                               mySession.loc[:,'waitingTime']>mySession.loc[:,'feedbackTime'])
mySession.loc[:,'isRewarded'] = np.logical_and(mySession.loc[:,'isRewarded'],
                                               np.logical_not(mySession.isCatch))
mySession.loc[:,'trialDur'] = mySession.loc[:,'waitingTime']
mySession.loc[mySession.loc[:,'isRewarded'],'trialDur'] = mySession.loc[mySession.loc[:,'isRewarded'],'feedbackTime']
rho = mySession.isRewarded.sum()/mySession.trialDur.sum()
mySession.loc[:,'Return'] = mySession.isRewarded - rho*mySession.trialDur
mySession.loc[:,'abs_stim'] = abs(50 - mySession.loc[:,'stim'])

X = [mySession.loc[:,var].values.astype(float) for var in ['perc','isChoiceLeft','waitingTime']]

#%% CRITIC
ndx = mySession.isChoiceLeft==mySession.isChoiceLeft
X = [mySession.loc[ndx,var].values.astype(float) for var in ['perc','isChoiceLeft','waitingTime']]
y = mySession.loc[ndx,'Return'].values

in_perc = kr.layers.Input(shape=(1,), name='perc')
in_ti = kr.layers.Input(shape=(1,), name='waitingTime')
# in_cho = kr.layers.Input(shape=(1,), name='isChoiceLeft')
state_action_pair = kr.layers.concatenate([in_perc,in_ti])
critic_h1 = kr.layers.Dense(units=200,activation='relu', name='critic_h1')(state_action_pair)
critic_h2 = kr.layers.Dense(units=200,activation='softmax', name='critic_h2')(critic_h1)
critic_h3 = kr.layers.Dense(units=50,activation='tanh', name='critic_h3')(critic_h2)
critic_h4 = kr.layers.Dense(units=50,activation='relu', name='critic_h4')(critic_h3)
critic_h5 = kr.layers.Dense(units=50,activation='tanh', name='critic_h5')(critic_h4)
critic_h6 = kr.layers.Dense(units=50,activation='relu', name='critic_h6')(critic_h5)
critic_output = kr.layers.Dense(units=1,activation='linear', name='critic_output')(critic_h6)
critic = kr.models.Model(inputs=[in_perc,in_ti], outputs=critic_output, name='critic')
critic.summary()
critic.compile(optimizer='Adam',loss='mean_squared_error')
# critic.fit(x=X,y=kr.utils.to_categorical(mySession.isChoiceLeft.values))
critic.fit(x=X,y=y,batch_size=1000)
#%%
y_hat = critic.predict(x=X)
mySession.loc[ndx,'hatReturn'] = y_hat

sns.pairplot(data=mySession.loc[ndx,:].sample(1000),hue='isRewarded',vars=['perc','isChoiceLeft','waitingTime','Return','hatReturn'])
plt.show()

#%
sns.regplot(data=mySession,x='Return',y='hatReturn',ci=None,x_bins=np.percentile(mySession.Return,np.linspace(0,100,1000)),fit_reg=False)
# sns.regplot(data=mySession,x='Return',y='hatReturn',ax=ha[1],ci=None)
# sns.regplot(data=mySession,x='Return',y='hatReturn',ax=ha[2],ci=None)
plt.show()

#%%

warg =


#%%
plt.scatter(mySession.Return.values,y_hat)
plt.show()

#%%

#%% ACTOR
if 'actor' not in locals():
    actor = kr.models.Sequential()
§1    actor.add(kr.layers.Dense(units=10,activation='relu',input_dim=1))
    actor.add(kr.layers.Dense(units=10,activation='relu'))
    actor.add(kr.layers.Dense(units=1, activation='sigmoid'))
actor.compile(optimizer='SGD',
              loss='mse')
actor.summary()