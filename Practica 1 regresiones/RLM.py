import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm

def optimizar_BW(x, y):
    X_opt=x[:,[0,1,2,3,4,5,6]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=y, exog=X_opt).fit()
    print(Regresion_OLS.summary())

    X_opt=x[:,[0,1,2,3,5,6]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=y, exog=X_opt).fit()
    print(Regresion_OLS.summary())

    X_opt=x[:,[1,2,3,5,6]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=y, exog=X_opt).fit()
    print(Regresion_OLS.summary())

    X_opt=x[:,[2,3,5,6]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=y, exog=X_opt).fit()
    print(Regresion_OLS.summary())

    X_opt=x[:,[3,5,6]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=y, exog=X_opt).fit()
    print(Regresion_OLS.summary())


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
    #normalizar
    sc=StandardScaler()
    X=sc.fit_transform(X)

    #optimizar
    optimizar_BW(X,Y)
    X=X[:,[0,1,2,3,5,6]]
    print('------------------------------------')
    print('X optimizada: \n',X)

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

    print("R^2 : ", r2_score(Y_test, ypred))
    print("R^2 ajustada: ", 1 - (1-r2_score(Y_test, ypred))*(len(Y)-1)/(len(Y)-X.shape[1]-1))
