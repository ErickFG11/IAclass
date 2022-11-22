from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.utils as ku
import keras_tuner as kt
from keras import optimizers

def construir_modelo(hp):
    n_hidden=hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons=hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer=hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer=="sgd":
        optimizer=optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer=optimizers.Adam(learning_rate=learning_rate)

    model=Sequential()
    model.add(Flatten())
    for _ in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    names = ['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA']
    numeroclases=7
    n_hidden=7
    n_neurons=124
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
    Y=ku.to_categorical(Y)
    print('------------------------------------')

    #escalar datos
    sc_X=StandardScaler()
    X=sc_X.fit_transform(X)
    print('Caracteristicas estandarizadas \n',X)
    print('------------------------------------')
    #separar conjunto de datos
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
    #modelo de red
    model=Sequential()
    model.add(Flatten())
    for _ in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(units=numeroclases, activation='softmax'))
    #compilar modelo
    random_search_tuner=kt.RandomSearch(construir_modelo, objective="val_accuracy", max_trials=20, seed=42,
                                    overwrite=True, project_name="frijoles")
    #ejecutar red neuronal
    random_search_tuner.search(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test))

    best_trial=random_search_tuner.get_best_hyperparameters(num_trials=3)
    print(best_trial[0].values)

