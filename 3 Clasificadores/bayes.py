import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Social_Network_Ads.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    #conjunto de caracteristicas
    X=dataset.iloc[:,2:4].values
    print('Caracteristicas \n',X)
    print('------------------------------------')
    #vector de clases
    Y=dataset.iloc[:,-1].values
    print('Clases \n',Y)
    print('------------------------------------')
    #reemplazar datos faltantes con la media de los demas datos
    imputer=SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
    imputer.fit(X[:,0:2])
    X[:,0:2]=imputer.transform(X[:,0:2])
    print('Caracteristicas preprocesadas \n',X)
    print('------------------------------------')
    #Codificando datos categoricos
    labelencoder_Y=LabelEncoder()
    Y=labelencoder_Y.fit_transform(Y)
    #escalar datos
    sc_X=StandardScaler()
    X=sc_X.fit_transform(X)
    print('Caracteristicas estandarizadas \n',X)
    print('------------------------------------')
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #entrenar naive bayes
    classifier=GaussianNB()
    classifier.fit(X_train, Y_train)

    #Graficar
    X_set, Y_set=X_train, Y_train
    X1, X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                       np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(Y_set)):
        plt.scatter(X_set[Y_set==j,0], X_set[Y_set==j,1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Naive bayes training set')
    plt.xlabel('Edad')
    plt.ylabel('Salario estimado')
    plt.legend()
    plt.show()
