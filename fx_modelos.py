from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import fx_resultados as res
from tensorflow import keras
from keras import layers, models, optimizers
import keras_tuner as kt
from sklearn.model_selection import train_test_split



#se prepara la validación cruzada de los datos
def validacion_cruzada(split):
    skf = StratifiedKFold(n_splits= split, shuffle=True, random_state=1)
    scoring = ['precision_macro', 'recall_macro', 'precision_micro', 'recall_micro', 'f1_macro','accuracy']
    return skf, scoring

#se define un modelo de regresión logaritmica con todos sus hiperparámetros
def hip_reg_log():
    modelo = LogisticRegression()
    penalty = ['l2']
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
    max_iters = [10000, 20000]
    grid = dict(penalty=penalty, solver=solvers, max_iter = max_iters)
    return modelo, grid

#se define un modelo de kn vecinos con todos sus hiperparámetros
def hip_kn_vecinos():
    modelo = KNeighborsClassifier()
    vecinos = [121, 61, 31, 5]
    algortimo = ['auto', 'ball_tree', 'kd_tree', 'brute']
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    grid = dict(n_neighbors=vecinos, weights=weights, algorithm = algortimo, metric=metric)
    return modelo, grid

#se define un modelo de SVM con todos sus hiperparámetros
def hip_svc():
    modelo = SVC(gamma='auto')
    kernels = ['poly', 'rbf', 'sigmoid']
    Cc = [50, 10, 1.0, 0.1, 0.01]
    gammas = ['scale']
    grid = dict(kernel=kernels, C=Cc, gamma=gammas)
    return modelo, grid

#se define un modelo de red neuronal profunda
def hip_red_neuronal(hp):
    modelo = keras.Sequential()
    for i in range(hp.Int('num_capas', 2, 10)):
        modelo.add(layers.Dense(units= hp.Int('capa_' + str(i), min_value = 10, max_value = 1000, step = 100), activation='relu'))
    modelo.add(layers.Dense(2, activation='softmax'))
    
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    opt = optimizers.adam_v2(learning_rate = hp_learning_rate)
    modelo.compile(loss='categorical_crossentropy', optimizer=opt, metrics='accuracy')
    return modelo

def optimizar_RN(matriz_dato, matriz_clase):
    tuner = kt.Hyperband(hip_red_neuronal, objective= 'val_accuracy', max_epochs= 10, factor= 3, directory = 'dir', project_name = 'x')
    X_train, X_test, y_train, y_test = train_test_split(matriz_dato, matriz_clase, test_size=0.3, random_state= True)
    tuner.search(X_train, y_train, epochs= 10, validation_data = (X_test, y_test))
    tuner.results_summary()


    

#se obtiene el mejor modelo posible variando los hiperparámetros
def optimizar_modelos(matriz_dato, matriz_clase, modelo, grid):
    df = res.crear_dataframe_metricas()
    skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(matriz_dato, matriz_clase):
        X_train, X_test = matriz_dato[train_index], matriz_dato[test_index]
        y_train, y_test = matriz_clase[train_index], matriz_clase[test_index]
        for i in range(0,len(modelo)):
            mod_optimo = GridSearchCV(estimator=modelo[i], param_grid=grid[i], n_jobs=-1, cv=skf, scoring='accuracy', error_score=0) 
            mod_optimo_ent = mod_optimo.fit(X_train, y_train)
            y_pred = mod_optimo.predict(X_test)
            df = obtener_resultados(i, mod_optimo_ent, y_test, y_pred, df)
    return df

#se encarga de ir llenando el dataframe seguyn el modelo en que se trabaje
def obtener_resultados(cont, mod_optimo_ent, y_test, y_pred, df):
    mejores_parametros = str(mod_optimo_ent.best_params_)
    result = res.metricas(y_test, y_pred)
    
    if cont == 0:
        tipo = 'Reg Log'
    elif cont == 1:
        tipo = 'KNN Vecinos'
    elif cont == 2:
        tipo = 'SVC'
    elif cont == 3:
        tipo = 'Red Neuronal'
    
    df = res.guardar_dato_dataframe_metricas(result, mejores_parametros, tipo, df)
    return df
