import pandas as pd
import numpy as np

def cor_aux(df,i,j,res):
    ''' 
    Cálculo de la correlación para un par de variables de un dataset.

    Args:
        df (DataFrame): Dataset de diferentes variables.
        i (int | str): Columna de una variable númerica.
        j (int | str): Columna de una variable númerica.
        res (DataFrame): Matriz de correlaciones a actualizar.

    Returns:
        None
    '''
    val_aux=sum((df[i] -np.mean(df[i])) * (df[j] - np.mean(df[j]))) / np.sqrt(sum((df[i] - np.mean(df[i]))**2) * sum((df[j] - np.mean(df[j]))**2))
    res.loc[i,[j]]=val_aux
    res.loc[j,[i]]=val_aux

def im_aux(df,i,j,res):
    ''' 
    Cálculo de la información mutua para un par de variables de un dataset.

    Args:
        df (DataFrame): Dataset de diferentes variables.
        i (int | str): Columna de una variable categórica.
        j (int | str): Columna de una variable categórica.
        res (DataFrame): Matriz de información mutua a actualizar.

    Returns:
        None
    '''
    x=df[i]
    y=df[j]
    n=len(x)
    pij=pd.Series(list(zip(x,y))).value_counts()/n
    pi=x.value_counts()/n
    pj=y.value_counts()/n

    def form(pij,pi,pj,index):
        i,j=index
        return pij*np.log(pij/(pi[i]*pj[j]))

    val_aux=sum(pd.Series(range(len(pij))).apply(lambda x,pi,pj:form(pij.iloc[x],pi,pj,pij.index[x]),pi=pi,pj=pj))
    res.loc[i,[j]]=val_aux
    res.loc[j,[i]]=val_aux

def cor_im(datos):
    ''' 
    Cálculo de la correlación/información mutua por pares entre variables de un dataset.

    Args:
        datos (DataFrame): Dataset con diferentes variables.

    Returns:
        (DataFrame): Dataset con las correlaciones/información mutua.
    '''
    res=pd.DataFrame([[0.0] * datos.shape[1]] * datos.shape[1])
    np.fill_diagonal(res.values, 1.0)
    res.index=datos.columns
    res.columns=datos.columns
    # Cálculo de las correlaciones para las variables númericas
    var_numb = datos.select_dtypes(include='number').columns
    for i in range(len(var_numb)-1):
        for j in range(i+1,len(var_numb)):
            cor_aux(datos,var_numb[i],var_numb[j],res)
    
    # Cálculo de la información mutua para las variables categóricas
    var_cat = datos.select_dtypes(exclude='number').columns
    for i in range(len(var_cat)):
        for j in range(i,len(var_cat)):
            im_aux(datos,var_cat[i],var_cat[j],res)

    return res 