import pandas as pd
from . import metricas

def filtrado(datos, clase=None,min_Var=float('-inf'),max_Var=float('inf'),min_AUC=0,max_AUC=1,min_Entropia=float('-inf'),max_Entropia=float('inf')):
    ''' 
    Filtrado de variables en base a las métricas implementadas.

    Args:
        datos (DataFrame): Dataset de diferentes variables.
        clase (int | str): Columna de la variable de la clase binaria, default=None.
        ... : Diferentes valores de umbrales para filtrar las variables.

    Returns:
        (DataFrame): Dataset filtrado.
    '''
    metricas=metricas.calcular_metricas(datos,clase)
    # Completar metricas
    metricas.loc['Varianza',~pd.notna(metricas.loc['Varianza'])]=min_Var
    metricas.loc['AUC',~pd.notna(metricas.loc['AUC'])]=min_AUC
    metricas.loc['Entropia',~pd.notna(metricas.loc['Entropia'])]=min_Entropia
    # Creación del filtro
    filtro=(metricas.loc['Varianza'] < min_Var) | (metricas.loc['Varianza'] > max_Var)
    filtro=filtro | (metricas.loc['AUC'] < min_AUC) | (metricas.loc['AUC'] > max_AUC)
    filtro=filtro | (metricas.loc['Entropia'] < min_Entropia) | (metricas.loc['Entropia'] > max_Entropia)
    # Borrado de columnas
    datos = datos.drop(columns = filtro[filtro==True].index)
    return datos