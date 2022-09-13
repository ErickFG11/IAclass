import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm

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

    X_opt=X[:,[0,1,2,3,4]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
    print(Regresion_OLS.summary())

    X_opt=X[:,[1,2,3,4]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
    print(Regresion_OLS.summary())

    X_opt=X[:,[2,3,4]]
    X_opt=np.array(X_opt, dtype=float)
    Regresion_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
    print(Regresion_OLS.summary())
