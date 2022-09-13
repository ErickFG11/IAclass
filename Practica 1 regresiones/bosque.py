import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    cols=["MPG", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
    #importar dataset reemplazando los ? por NaN y quitando la variable categorica
    dataset=pd.read_csv("auto-mpg.data", na_values="?", comment='\t', sep=' ', skipinitialspace=True, names=cols)
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,1:].values
    Y=dataset.iloc[:,0].values
    print('Caracteristicas: \n',X)
    print('------------------------------------')
    print('Clases Y: \n',Y)

    #reemplazar datos faltantes con la media de los demas datos
    imputer=SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
    imputer.fit(X[:, :])
    X[:, :]=imputer.transform(X[:, :])

    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #arbol de desición para regresion
    regresion=RandomForestRegressor(n_estimators=10 ,random_state=0) #numero de arboles 10
    regresion.fit(X_train, Y_train)

    #predicciones
    ypred=regresion.predict(X_test)

    print('------------------------------------')
    print("R^2 : ", r2_score(Y_test, ypred))
    print("R^2 ajustada: ", 1 - (1-r2_score(Y_test, ypred))*(len(Y)-1)/(len(Y)-X.shape[1]-1))
    '''
    X_grid=np.arange(min(X), max(X), 0.01)
    X_grid=X_grid.reshape((len(X_grid),1))
    plt.scatter(X,Y,color='red')
    plt.plot(X_grid, regresion.predict(X_grid), color='blue')
    plt.title('Regresion con bosques aleatorios')
    plt.xlabel('Posicion')
    plt.ylabel('Salario')
    plt.show()
    '''
