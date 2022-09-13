import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
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


if __name__ == '__main__':
    cols=["MPG", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
    #importar dataset
    dataset=pd.read_csv("auto-mpg.data", na_values="?", comment='\t', sep=' ', skipinitialspace=True, names=cols)
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    #X=dataset.iloc[:,1:].values
    X=dataset.iloc[:,1:].values
    Y=dataset.iloc[:,0].values
    print('Caracteristicas: \n',X)
    print('------------------------------------')
    print('Clases Y: \n',Y)

    #reemplazar datos faltantes con la media de los demas datos
    imputer=SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
    imputer.fit(X[:, :])
    X[:, :]=imputer.transform(X[:, :])
    optimizar_BW(X,Y)


