from setuptools import setup

setup(
   name='SME.Xabi',
   version='0.0.1',
   author='Xabier Larrayoz',
   author_email='xabier.larrayoz@ehu.eus',
   packages=['SME.Xabi'],
   package_dir={'SME.Xabi':'src'},
   url='https://github.com/Fausro/SME.Xabi.Python',
   description='Paquete de funciones de SME de Python. Este paquete implementa las funciones que se solicitaban y algunas funcionalidades adicionales.',
   install_requires=[
      "seaborn",
      "pandas",
      "matplotlib",
      "numpy"
   ],
)