import pandas as pd
import numpy as np

def normalizacion(x):
    ''' 
    Normaliza el vector x, con valores entre 0 y 1.

    Args:
        x (ndarray): Vector númerico.

    Returns:
        (ndarray): Vector normalizado.
    '''
    return (x-min(x))/(max(x)-min(x))

def estandarizacion(x):
    ''' 
    Estandariza el vector x, con media 0 y desviación 1.

    Args:
        x (ndarray): Vector númerico.

    Returns:
        (ndarray): Vector estandarizado.
    '''
    return (x-np.mean(x))/np.std(x)

def scale_x(x,norm=False):
    ''' 
    Normalización y estandarización de variables tanto para un dataset como de manera individual.

    Args:
        x (DataFrame | ndarray): DataFrame o vector númerico para escalar.
        norm (bool): Booleano que determina el tipo de transformación a realizar, default=False.
            + True: estandariza
            + False: normaliza

    Returns:
        (DataFrame | ndarray): Entrada transformada. 
    '''
    if isinstance(x, pd.DataFrame): # Si df es un data.frame
        var_number = x.select_dtypes(include='number').columns # Solo se aplica a las variables númericas
        if norm:
            x[var_number]=x[var_number].apply(normalizacion)
        else:
            x[var_number]=x[var_number].apply(estandarizacion)
        return x
    else:
        if norm:
            return normalizacion(x)
        else:
            return estandarizacion(x)