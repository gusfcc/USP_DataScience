#Gustavo Fernandes Carneiro de Castro - 11369684


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

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


gnb = GaussianNB()
crossValYPred = cross_val_predict(gnb, X, Y, cv = 10)

confMatrix = confusion_matrix(Y,crossValYPred,labels=range(0,10) ) #labels=["C","E","HM","J","P","R","RAP","ReB","RG","S"]
confDisplay = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels=Classes)
confDisplay.plot()
plt.show()