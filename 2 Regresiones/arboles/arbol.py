import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Salaries.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,1:-1].values
    Y=dataset.iloc[:,-1].values

    #arbol de desición para regresion
    regresion=DecisionTreeRegressor(random_state=0)
    regresion.fit(X,Y)

    print('Prediccion 9.5\n',regresion.predict([[9.5]]))

    X_grid=np.arange(min(X), max(X), 0.01)
    X_grid=X_grid.reshape((len(X_grid),1))
    plt.scatter(X,Y,color='red')
    plt.plot(X_grid, regresion.predict(X_grid), color='blue')
    plt.title('Regresion con arboles de desicion')
    plt.xlabel('Posicion')
    plt.ylabel('Salario')
    plt.show()
