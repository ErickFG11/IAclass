import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

if __name__ == '__main__':
    #importar dataset
    dataset=pd.read_csv("Dry_Bean_Dataset.csv")
    print('Dataset Original \n',dataset)
    print('------------------------------------')
    #conjunto de caracteristicas
    X=dataset.iloc[:,:-1].values
    print('Caracteristicas \n',X)
    print('------------------------------------')
    #vector de clases
    Y=dataset.iloc[:,-1].values
    print('Clases \n',Y)
    print('------------------------------------')
    #Codificando datos categoricos
    labelencoder_Y=LabelEncoder()
    Y=labelencoder_Y.fit_transform(Y)
    print('Clases codificadas\n', Y)
    print('------------------------------------')

    #escalar datos
    sc_X=StandardScaler()
    X=sc_X.fit_transform(X)
    print('Caracteristicas estandarizadas \n',X)
    print('------------------------------------')
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #entrenar bosque
    classifier=RandomForestClassifier(n_estimators=10, random_state=0)
    classifier.fit(X_train, Y_train)

    #Predict the response for test dataset
    y_pred = classifier.predict(X_test)
    accu=accuracy_score(Y_test, y_pred)
    print("Model Accuracy: {:.2f}%".format(accu * 100))
    #matriz de confusion
    cm=confusion_matrix(Y_test, y_pred)
    plot_confusion_matrix(classifier, X_test, Y_test)
    plt.show()
