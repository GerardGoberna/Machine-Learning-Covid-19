import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#Llegim les dades
dataset2 = pd.read_excel("C:/Users/Gerard/Desktop/TFM/Casos Covid3.xls", index_col="Data", parse_dates=["Data"])


#Separem entre set d'entrenament i de test

trainingSet = dataset2.iloc[0:214,:]
testSet = dataset2.iloc[214:310,:]

#Ingressos
dataset2 = pd.read_excel("C:/Users/Gerard/Desktop/TFM/ingressos.xls", index_col="Data", parse_dates=["Data"])
#Partició dels dos sets pel cas d'ingressos
trainingSet = dataset2.iloc[0:186,:]
testSet = dataset2.iloc[186:249,:]

#Normalitzem els valors de les dades d'entrenament
sc = MinMaxScaler(feature_range=(0, 1))
trainingSetScaled = sc.fit_transform(trainingSet)


#Entrenarem la xarxa proporcionant 14 dades d'entrada i un de sortida en cada interacció
timeSteps = 14
xTrain = []
yTrain = []

#xTrain = llista de conjunts de 14 dades.
#yTrain = llista de valors

for i in range(0, len(trainingSetScaled) - timeSteps):
    xTrain.append(trainingSetScaled[i:i + timeSteps, 0])
    yTrain.append(trainingSetScaled[i + timeSteps, 0])

#Utilitzarem la llibreria numpy ja que haurem d'afegir una dimensió a xTrain'
xTrain, yTrain = np.array(xTrain), np.array(yTrain)

#Afegim la dimensió
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))


#Aquests paràmetres els entrarem a Keras, i són els que hem escollit òptims per a la realització de la xarxa
dim_entrada = (xTrain.shape[1], 1)
dim_salida = 1
na = 100


#Iniciem el model i afegim les capes, que per aquest cas hem cregut que amb una capa LSTM i una capa de sortida en tindrem prou,
# ja que amb més capes obteníem resultats semblants o pitjors
regresor = Sequential()

#Primera i última capa
regresor.add(LSTM(units=na, input_shape=dim_entrada))

#Capa output
regresor.add(Dense(units=dim_salida))

regresor.summary()

regresor.compile(optimizer='rmsprop', loss='mse')


#Utilitzarem l'objecte callbacks per poder estimar millor els paràmetres amb els que entrenarem el set d'entrenament,
# i els paràmetres que hem posat són els òptims

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)

#Comprovem el epoch òptim per perdre el mínim

regresor.fit(xTrain, yTrain, epochs=50, batch_size=50, callbacks=[earlystopping])

regresor.fit(xTrain, yTrain, epochs=25, batch_size=64)


#Normalitzem el conjunt de test i fem el mateix que amb el d'entrenament
auxTest = sc.transform(testSet.values)
xTest = []

for i in range(0, len(auxTest) - timeSteps):
    xTest.append(auxTest[i:i + timeSteps, 0])



xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))


#Realitzem la predicció
prediccion = regresor.predict(xTest)


#Desnormalitzem els valors de la predicció per obtenir valors normals
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados


def visualizar(real, prediccion):
    plt.plot(real[0:len(prediccion)], color='red', label='Nombre de casos Covid')
    plt.plot(prediccion, color='blue', label='Predicció Casos')
    plt.xlabel('Temps')
    plt.ylabel('Casos')
    plt.legend()
    plt.show()

visualizar(testSet.values, prediccion)

Taula_parametres = pd.DataFrame({
    'Paràmetres': ['Nombre de capes', 'Nombre de neurones per capa', 'Time steps', 'Epochs', 'Batch sice'],

    'Valors': ['1 LSTM i una de sortida', 100, 14,
              6, 64]})
print(Taula_parametres)