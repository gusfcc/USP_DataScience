#Gustavo Fernandes Carneiro de Castro - 11369684


import warnings

from numpy.typing import _128Bit
warnings.filterwarnings('ignore')

import librosa
import librosa.display
import numpy as np
import os
import pandas as pd

arrayMusicas = np.zeros([205,20]) #inicializa arrays que guardaram dados das músicas.
arrayNomes = []
arrayGenres = []

i = 0
for file in os.scandir("musicas205"): #itera pelas músicas dentro da pasta

    #Pega o arquivo, e manipula seu nome e gênero para salva-lo.
    fFullPath = file.path
    audio_name = os.path.split(os.path.splitext(fFullPath)[0])[1]

    genre = audio_name.split("_")[0]
    nome = audio_name.split("_")[1]
    arrayGenres.append(genre)
    arrayNomes.append(nome)

    #utiliza librosa para obtenção dos coeficientes nos tempos t
    x , sr = librosa.load(fFullPath)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    #itera pela array, fazendo média do coeficiente para cada música.
    j = 0
    for t in mfccs:
        arrayMusicas[i,j] = np.mean(t)
        j += 1
 
    i += 1

#Cria arquivo de database
dfMusicas= pd.concat([pd.DataFrame(arrayNomes),pd.DataFrame(arrayMusicas),pd.DataFrame(arrayGenres)],1)
print(dfMusicas)

csv = pd.DataFrame.to_csv(dfMusicas,index=False)
file = open("mfcc.txt",'w')

file.write(csv)
file.close()