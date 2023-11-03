import pandas as pd
import numpy as np

def calc_entropia(x):
    '''
    Calcula la entropía de de una variable discretizada.

    Args:
        x (AnyArrayLike | list | tuple): Vector de valores discretizados.

    Returns:
        (float): Entropía del vector.
    '''
    frecuencias = x.value_counts()
    probabilidades = frecuencias/len(x)
    probabilidades = probabilidades[probabilidades>0] # Evitar problemas de NA
    return -sum(probabilidades*np.log(probabilidades))

def calc_integral(x, y):
    '''
    Calcula el área que queda por debajo de una linea de puntos.

    Args:
        x (ArrayLike): Vector de valores de la variable independiente.
        y (ArrayLike): Vector de valores de la variable dependiente.

    Returns:
        (float): Entropia del vector.
    '''
    delta_x = np.diff(x)
    a=np.delete(y,0) # Para quietar los 0s del principio 
    b=np.delete(y,len(y)-1) # y del final
    mean_y=np.mean(np.column_stack((a,b)), axis=1)
    return (abs(np.sum(np.dot(delta_x,mean_y),axis=0)))

def calc_auc(dx,dy):
    '''
    Calcula el AUC, TPR y FPR.

    Args:
        dx (Serie): Vector de valores de una variable continua.
        dy (Serie): Vector de etiquetas.

    Returns:
        (float): Valor de AUC.
        (list): True Positive Rate.
        (list): False Positive Rate.
    '''
    if dy.dtype=='O': # Si las clases se han definido por '1' y '0'
        dy=dy.astype(int).astype(bool)
    # Valores de corte, que dan lugar a cambios en el TPR y FPR
    ptos_corte=sorted(dx.unique())

    tpr_arr, fpr_arr = [1], [1]
    for p_corte in ptos_corte: # Todas las predicciones posibles
        predict=dx>=p_corte 
        tp=sum(predict & dy)
        fp=sum(predict & ~dy)
        tn=sum(~predict & ~dy)
        fn=sum(~predict & dy)
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)
    tpr_arr.append(0)
    fpr_arr.append(0)

    auc=calc_integral(fpr_arr,tpr_arr)
    return auc,tpr_arr,fpr_arr

def calc_var(x):
    ''' 
    Calcula la varianza.

    Args:
        x (Array): Vector de valores numéricos.

    Returns:
        (float): Valor de la varianza.
    '''
    return np.sum((x-np.mean(x,axis=0))**2,axis=0)/(len(x)-1)

def calcular_metricas(datos, clase=None):
    ''' 
    Cálculo de métricas para los atributos de un dataset.

    Args:
        datos (DataFrame): Dataset con diferentes variables.
        clase (int | str): Columna de la variable de la clase binaria, default=None.

    Returns:
        (DataFrame): Dataset con la varianza, AUC, y Entropia correspondiente a cada columna de datos.
    '''
    res = pd.DataFrame([[None] * datos.shape[1]] * 3)
    res.columns=datos.columns
    res.index=["Varianza", "AUC", "Entropia"]
    # Variables continuas
    var_num = datos.select_dtypes(exclude='object').columns
    if var_num.size > 0:
        res.loc['Varianza',var_num]=calc_var(datos[var_num])
        if clase: # Si se ha especificado una variable binaria
            res.loc['AUC',var_num]=datos[var_num].apply(calc_auc,dy=datos[clase]).iloc[0]
    # Variables discretas
    var_discretas = datos.select_dtypes(include='object').columns
    if var_discretas.size > 0:
        res.loc['Entropia',var_discretas] = datos[var_discretas].apply(calc_entropia).iloc[0]
    return res