from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.utils as ku
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import keras_tuner as kt

def modelo(hp):


if __name__ == '__main__':
    names = ['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA']
    numeroclases=7
    #importar dataset
    dataset=pd.read_csv("Dry_Bean_Dataset.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    #conjunto de caracteristicas
    X=dataset.iloc[:,:-1].values
    print('Caracteristicas \n',X)
    print('------------------------------------')
    #vector de clases
    Y=dataset.iloc[:,-1].values
    print('Clases \n',Y)
    print('------------------------------------')

    #Codificando datos categoricos
    labelencoder_Y=LabelEncoder()
    Y=labelencoder_Y.fit_transform(Y)
    print('Clases codificadas\n', Y)
    Y=ku.to_categorical(Y)
    print('------------------------------------')

    #escalar datos
    sc_X=StandardScaler()
    X=sc_X.fit_transform(X)
    print('Caracteristicas estandarizadas \n',X)
    print('------------------------------------')
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #modelo de red
    model=Sequential()
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=numeroclases, activation='softmax'))
    #compilar modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #ejecutar red neuronal
    model.fit(X_train, Y_train, epochs=100, batch_size=32)

    #Prediccion
    y_pred = model.predict(X_test)
    predicciones=[]
    reales=[]
    for i in range(len(y_pred)):
        predicciones.append(names[np.argmax(y_pred[i])])
        reales.append(names[np.argmax(Y_test[i])])

    accu=accuracy_score(reales, predicciones)
    print("Model Accuracy: {:.2f}%".format(accu * 100))
    #matriz de confusion
    cm=confusion_matrix(reales, predicciones)
    print(cm)


