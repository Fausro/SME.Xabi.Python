import pandas as pd
import numpy as np

def discretize(x, num_bins, algoritmo='anchura'):
    '''
    Realiza la discretización tanto para un solo atributo como para un dataset completo.

    Args:
        x (DataFrame | list | ndarray | Series): Un DataFrame, list, ndarray o Series que será discretizado.
        num_bins (int): Un número natural mayor de 1, que determina el número de intervalos.
        algoritmo (str): Algoritmo de discretización, default='anchura'.
            + Implementados: anchura, frecuencia, cuantil

    Returns:
        (DataFrame | ndarray): DataFrame o vector de valores categóricos resultado de aplicar el algoritmo.

    Raises:
        ValueError: 
            + Si el algoritmo indicado no es uno de los disponibles.
            + Si el número de intervalos no es adecuado.

    Example:
        >>> discretize([1,2,3,4],2)
        ['(-inf, 2.5]', '(-inf, 2.5]', '(2.5, inf)', '(2.5, inf)']
    '''
    # Control básico de errores
    if algoritmo not in ['anchura','frecuencia','cuantil']: 
        raise ValueError("El algoritmo indicado no es válido.")
    elif not (num_bins>1 and type(num_bins)==int):
        raise ValueError("El num_bins tiene que ser un número natural mayor de 1.")

    if isinstance(x, pd.DataFrame): # Si x es un data.frame
        # Realizar la discretización para cada una de las columnas
        return x.apply(discretize,num_bins=num_bins,algoritmo=algoritmo) 
    else: # x es un atributo
        if not isinstance(x,np.ndarray):
            x=np.array(x)
        if algoritmo=='anchura':
            minimo = min(x)
            w = (max(x)-minimo)/num_bins # Tamaño de los intervalos

            # Hay num_bins-1 puntos de corte, y cada uno será el anterior más w
            ptos_corte=np.arange(1,num_bins)
            ptos_corte=ptos_corte*w+minimo
        else: # Algoritmo por frecuencia o cuantil
            orden=np.argsort(x) # Es necesario el orden para determinar los puntos de corte 
            n=len(x)
            indices=np.arange(1,num_bins)
            indices=(indices*n)/num_bins
            indices=np.where(indices % 1 != 0, indices.astype(int), (indices - 1).astype(int))
            if algoritmo=='frecuencia':
                # El punto de corte de dos intervalos consecutivos será el punto medio entre los dos elementos más cercanos de dichos intervalos
                ptos_corte=(x[orden[indices]]+x[orden[indices+1]])/2
            elif algoritmo=='cuantil':
                # El punto de corte será el cuartil correspondiente
                ptos_corte=x[orden[indices]]  

        # Vector de valores categóricos
        etiquetas = np.repeat(None, len(x)) # Para evitar problemas con la longitud de los strings
        etiquetas[:] = '({}, inf)'.format(ptos_corte[-1])

        aux=float('-Inf')
        for p_corte in ptos_corte: # Asignación de los elementos al intervalo correspondiente
            etiquetas[(aux<x) & (x<=p_corte)]='({}, {}]'.format(aux,p_corte)
            aux=p_corte
        return etiquetas