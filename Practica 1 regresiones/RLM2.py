import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

def dommies(x):
    #Codificando datos categoricos
    labelencoder_X=LabelEncoder()
    x[:,7]=labelencoder_X.fit_transform(x[:,7])
    print('------------------------------------')
    print('Caracteristicas dummy \n',x)
    df = pd.DataFrame(x)
    df.to_csv('raw_data.csv', index=False)
    #Traducir categoricos a numeros
    #ct=ColumnTransformer([("Country", OneHotEncoder(), [7])], remainder="passthrough")
    #x=ct.fit_transform(x)
    #print('Categoricos a numeros \n',x)

if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("auto-mpg.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    X=dataset.iloc[:,1:].values
    Y=dataset.iloc[:,0].values
    print('Caracteristicas: \n',X)
    print('------------------------------------')
    print('Clases Y: \n',Y)

    dommies(X)
