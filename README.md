# SME.Xabi.Python

Paquete de funciones de SME de Python. Este paquete implementa las funciones que se solicitaban y algunas funcionalidades adicionales.

## Instalación

Puedes instalar el paquete utilizando pip:

```bash
pip install git+https://github.com/Fausro/SME.Xabi.Python.git
```

#### Dependencias

- pandas
- numpy
- matplotlib
- seaborn

## Uso

A continuación, se muestra un ejemplo de cómo utilizar el paquete:

```python
from SME.Xabi.discretizacion import discretize

print(discretize([1,2,3,4],2))
# ['(-inf, 2.5]' '(-inf, 2.5]' '(2.5, inf)' '(2.5, inf)']
```

## Contacto

Xabier Larrayoz (<xabier.larrayoz@ehu.eus>)
