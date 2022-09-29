import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Salary_Data.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,0:-1].values #años de experiencia
    Y=dataset.iloc[:,-1].values #Salario
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    regresion=LinearRegression()
    regresion.fit(X_train, Y_train)
    ypred=regresion.predict(X_test)
    print(ypred)

    #visualizacion de datos de entrenamiento
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, regresion.predict(X_train), color='blue')
    plt.title('Salario vs Experiencia (set entrenamiento)')
    plt.xlabel('Años de experiencia')
    plt.ylabel('Salario')
    plt.grid()
    plt.show()

    #visualizacion de resultados
    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_train, regresion.predict(X_train), color='blue')
    plt.title('Salario vs Experiencia (set prueba)')
    plt.xlabel('Años de experiencia')
    plt.ylabel('Salario')
    plt.grid()
    plt.show()
