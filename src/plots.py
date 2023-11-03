import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc(tpr, fpr):
    """    
    Función para trazar la curva ROC.

    Args:
        tpr (list): Lista de tasas de verdaderos positivos.
        fpr (list): Lista de tasas de falsos positivos.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC')
    plt.fill_between(fpr, tpr, color='blue', alpha=0.2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

def plot_cor_im(df_cor_im):
    """    
    Función para la representación de correlación/información mutua.

    Args:
        df_cor_im (DataFrame): Dataset con las correlaciones e información mutua.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    sns.heatmap(df_cor_im, annot=True, cmap='coolwarm', linewidths=0.5, cbar=True)
    plt.title('Matriz de Correlación e información mutua', fontsize=14)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.show()