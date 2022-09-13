import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Salaries.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,1:-1].values
    Y=dataset.iloc[:,-1].values
    #regresion lineal
    Lin_reg=LinearRegression()
    Lin_reg.fit(X,Y)

    #regresion polinomial
    poly_reg=PolynomialFeatures(degree=5)#numero de terminos
    X_poly=poly_reg.fit_transform(X)
    lin_reg2=LinearRegression()
    lin_reg2.fit(X_poly,Y)

    #Visualizacion RL
    plt.scatter(X, Y, color='red')
    plt.plot(X, Lin_reg.predict(X), color='blue')
    plt.title('Regresion lineal')
    plt.xlabel('Posicion')
    plt.ylabel('Salario')
    plt.grid()
    plt.show()

    #visualizacion de resultados
    plt.scatter(X, Y, color='red')
    plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Regresion polinomial')
    plt.xlabel('Posicion')
    plt.ylabel('Salario')
    plt.grid()
    plt.show()


