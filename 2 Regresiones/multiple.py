import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("50_Startups.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,:-1].values
    Y=dataset.iloc[:,-1].values
    print('Caracteristicas: \n',X)
    print('------------------------------------')
    print('Clases Y: \n',Y)
    #Codificando datos categoricos
    ct=ColumnTransformer(transformers=[('enconder', OneHotEncoder(), [3])], remainder='passthrough')
    X=np.array(ct.fit_transform(X))
    print('------------------------------------')
    print('Variables dommies\n',X)
    #evitar trampa de variables ficticias
    X=X[:, 1:]
    print('------------------------------------')
    print('Quitar variables ficticias \n',X)
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #entrenamiento
    regresion=LinearRegression()
    regresion.fit(X_train, Y_train)
    #predicciones
    ypred=regresion.predict(X_test)

    print('\n Y de prueba\n', Y_test)
    print('------------------------------------')
    print('\n Predicciones\n', ypred)
    print('------------------------------------')
