#Gustavo Fernandes Carneiro de Castro - 11369684


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


# Faz com que os resultados sejam mantidos sempre que executar esse jupyter notebook
#SET_SEED = 56
#np.random.seed(SET_SEED)

import warnings
warnings.filterwarnings('ignore')

#Pega dados do dataBase
dfMusicas = pd.read_table("mfcc.txt", sep = ",", encoding = "ISO-8859-1", index_col=0)

X = dfMusicas.iloc[:,0:20].copy()
Y = dfMusicas.iloc[:,20].copy()
Classes = np.unique(Y)

#Normaliza os dados
scaler = MinMaxScaler()
lEncoder = LabelEncoder()
XNorm = scaler.fit_transform(X)
YNum = lEncoder.fit_transform(Y)

#Cria opções de MLP para ser testado pelo GridSearchCV()
mlp = MLPClassifier()
param_grid = [{'hidden_layer_sizes': [(9,9,9,9,9,9,9,9,9)], 'solver': ['adam', 'sgd', 'lbfgs'], 'activation': ['logistic', 'tanh', 'relu', 'identity'], 
            'learning_rate': ['constant', 'invscaling', 'adaptative'], 'learning_rate_init': [0.01,0.05,0.1]
            }]

#Testa as possiveis combinações.
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')
grid_search.fit(XNorm, YNum)
mlp = grid_search.best_estimator_

#utiliza crossValidation
crossValYPred = cross_val_predict(mlp,XNorm,YNum,cv=5)

#Cria e plota matriz de confusão
confMatrix = confusion_matrix(YNum,crossValYPred,labels=range(0,10) ) #labels=["C","E","HM","J","P","R","RAP","ReB","RG","S"]
confDisplay = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels=Classes)
confDisplay.plot()
plt.show()
