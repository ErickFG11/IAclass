import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Data.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    #conjunto de caracteristicas
    X=dataset.iloc[:,0:3].values
    #vector de clases
    Y=dataset.iloc[:,3].values
    #reemplazar datos faltantes con la media de los demas datos
    imputer=SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
    imputer.fit(X[:, 1:3])
    X[:, 1:3]=imputer.transform(X[:, 1:3])
    print('Caracteristicas preprocesadas \n',X)

    #Codificando datos categoricos
    labelencoder_X=LabelEncoder()
    labelencoder_Y=LabelEncoder()
    X[:,0]=labelencoder_X.fit_transform(X[:,0])
    Y=labelencoder_Y.fit_transform(Y)
    print('------------------------------------')
    print('Caracteristicas dummy \n',X)
    #Traducir categoricos a numeros
    ct=ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
    X=ct.fit_transform(X)
    print('Categoricos a numeros \n',X)
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #escalar datos
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)
    print('\n',X_train)
