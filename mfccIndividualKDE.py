#Gustavo Fernandes Carneiro de Castro - 11369684


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

p = 5 #divisão do KFold

#Puxar o dataset
dfMusicas = pd.read_table("mfcc.txt", sep = ",", encoding = "ISO-8859-1", index_col=0)

X = dfMusicas.iloc[:,:20].copy() # Dados X
YL = dfMusicas.iloc[:,20].copy() # Classes Y (com label string)
Classes = np.unique(YL)


#Normaliza os dados / transforma labels em numericos
#labels=["C","E","HM","J","P","R","RAP","ReB","RG","S"]
#labels=[ 0,  1,  2,   3,  4,  5,   6,    7,   8,   9 ]
scaler = MinMaxScaler()
lEncoder = LabelEncoder()
X = scaler.fit_transform(X)
Y = lEncoder.fit_transform(YL)

#criar vetor de Y preditos pelo algortimo
Ypred = np.full(len(Y), -1)

#Criação da malha
X_D = np.linspace(np.zeros(20),np.ones(20),500)

# Alisamento individual para cada atributo  -> ajeitar o bandwidth na mão
#                                            -> testar o algoritmo/plot
for i in range(1,20):
    kde = KernelDensity()
    param_grid = [{'bandwidth': np.arange(0.001,1.001,0.001), 
                'kernel': ['gaussian', 'tophat', 'epanechnikov', 
                                'exponential', 'linear', 'cosine']}]

    #Testa as possiveis combinações.
    grid_search = GridSearchCV(kde, param_grid)
    grid_search.fit(X, Y)
    kde = grid_search.best_estimator_
    
    kde.fit(X[:,i-1:i])
    log_density = kde.score_samples(X_D[:,i-1:i])
    plt.fill_between(X_D[:,i-1], np.exp(log_density), alpha=0.5)
    plt.plot(X[:,i-1], np.full_like(X[:,i-1], -0.01), '|k', markeredgewidth=1)
    #plt.ylim(-0.02, 1)
plt.show()
