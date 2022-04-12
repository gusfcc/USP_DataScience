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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from skimage.feature import peak_local_max

import warnings
warnings.filterwarnings('ignore')

p = 5 #divisão do KFold

#Puxar o dataset
dfMusicas = pd.read_table("mfcc.txt", sep = ",", encoding = "ISO-8859-1", index_col=0)

X = dfMusicas.iloc[:,:20].copy() # Dados X
YL = dfMusicas.iloc[:,20].copy() # Classes Y (com label string)
Classes = np.unique(YL)

#reduz o número de dimenções para 2 - facilitar plot
#todos os atributos são coeficiêntes na mesma escala
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#Normaliza os dados / transforma labels em numericos
#labels=["C","E","HM","J","P","R","RAP","ReB","RG","S"]
#labels=[ 0,  1,  2,   3,  4,  5,   6,    7,   8,   9 ]
scaler = MinMaxScaler()
lEncoder = LabelEncoder()
X = scaler.fit_transform(X)
Y = lEncoder.fit_transform(YL)

#criar vetor de Y preditos pelo algortimo
Ypred = np.full(len(Y), -1)

#criação do grid para a pdf
xx, yy = np.mgrid[0:1:500j,0:1:500j]
X_D = np.vstack([xx.ravel(), yy.ravel()]).T


#escolha automática de parâmetros
#se basea no estimador de maxima verossimilhança
kde = KernelDensity()
param_grid = [{'bandwidth': np.arange(0.001,1.001,0.001), 
                'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']}]

#Testa as possiveis combinações.
#grid_search = GridSearchCV(kde, param_grid, cv = p)
#grid_search.fit(X, Y)

#criar a kde (kernel utilizado e tamanho do espaçamento)
#kde = grid_search.best_estimator_ #0.068

#kde rapida para testes
kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.026) #0.042

#k_fold para o cross_val
kf = KFold(p, shuffle = True)

counter = 0 #contador auxiliar para o Ypred
check = 0 #contador de acertos

for train, test in kf.split(X):
    
    Xtrain = X[train] #X treino
    Xtest = X[test] #X teste
    Ytrain = Y[train] #Labels treino
    Ytest = Y[test]  #Labels teste
    
    kde.fit(Xtrain) #treinar kde
    
    #pegar a densidade para o grid
    log_density_X_D = kde.score_samples(X_D)
    
    #pegar a densidade para o Xtest
    log_density_test = kde.score_samples(Xtest)
    
    
    #Criar vetor dos picos da pdf 
    peak_local = peak_local_max(np.reshape(log_density_X_D, (500,500)))    
    density_maximas = np.zeros(len(peak_local), dtype = int)
    
    for i in range(len(peak_local)):
        density_maximas[i] = int(peak_local[i,0]*500 + peak_local[i,1])
    
    
    #Criar vetor para atribuir uma classe a cada pico da pdf
    maximas_class = np.full((len(density_maximas),10), 0)   
    maximas_class = np.c_[maximas_class, np.full(len(density_maximas),1)]
    
    
    #Ir de Xtreino em Xtreino verificando qual o pico mais próximo
    # e adicionar a classe do Xtreino analisado ao contador de classes por pico
    for i in range(len(Xtrain)):       
        idx = (np.sum((Xtrain[i] - X_D[density_maximas])**2, axis = 1).argmin())
        maximas_class[idx, Ytrain[i]] += 1
    
    #pegar a classe que mais apareceu de cada pico   
    maximas_labels = (np.argmax(maximas_class, axis = 1))
    
    
    #Ir de Xteste em Xteste verificando o pico correspondente.
    #Verificando se a classe atribuida ao pico é a mesma do Xteste
    for i in range(len(Xtest)):
        idx = ((log_density_X_D[density_maximas] - log_density_test[i])**2).argmin()           
        
        if(maximas_labels[idx] == Ytest[i]):
            check += 1
        
        Ypred[i + counter] = maximas_labels[idx] #add pred em Ypred
        
        #plotar pontos máximos encontrados
        plt.plot(X_D[density_maximas[idx],0], X_D[density_maximas[idx],1], marker = '.', color = 'crimson')
        
    #plot 
    plt.pcolormesh(xx, yy, np.reshape(np.exp(log_density_X_D),yy.shape))
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=2, facecolor='white')
    plt.scatter(Xtest[:,0], Xtest[:,1], s=2, facecolor='black')
    plt.show()
    
    counter += len(Ytest) #counter pro Ypred


#construção da matriz de confusão e accuracy
confMatrix = confusion_matrix(Y, Ypred, labels=range(0,10))
confDisplay = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels=Classes)
confDisplay.plot()
plt.show()
print(check)
print(check/len(Y))
