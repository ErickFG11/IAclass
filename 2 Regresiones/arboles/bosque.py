import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Salaries.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,1:-1].values
    Y=dataset.iloc[:,-1].values

    #arbol de desición para regresion
    regresion=RandomForestRegressor(n_estimators=10 ,random_state=0) #numero de arboles 10
    regresion.fit(X,Y)

    print('Prediccion 6.5\n',regresion.predict([[6.5]]))

    X_grid=np.arange(min(X), max(X), 0.01)
    X_grid=X_grid.reshape((len(X_grid),1))
    plt.scatter(X,Y,color='red')
    plt.plot(X_grid, regresion.predict(X_grid), color='blue')
    plt.title('Regresion con bosques aleatorios')
    plt.xlabel('Posicion')
    plt.ylabel('Salario')
    plt.show()
