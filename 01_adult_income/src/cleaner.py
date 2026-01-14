import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ruta_final = os.path.join(BASE_DIR, 'data', 'adult', 'adult.data')

print(f"Buscando archivo en: {ruta_final}")

nombres_columnas = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv(ruta_final, names=nombres_columnas, na_values='?', skipinitialspace=True)
print(f"Total de registros: {df.shape[0]}")
print("\nColumnas con valores faltantes (NaN):")
print(df.isnull().sum()[df.isnull().sum() > 0])
#Configurar el limpiador con estrategia mas frecuente
imputer = SimpleImputer(strategy='most_frequent')
#Columnas con nulos
columnas_con_nulos = ['workclass', 'occupation', 'native-country']
#Ejecutar la limpieza
df[columnas_con_nulos] = imputer.fit_transform(df[columnas_con_nulos])
#Comprobación final
print("\nVerificacion tras Limpieza")
print(f"Total de nulos en el dataset: {df.isnull().sum().sum()}")
print(df[columnas_con_nulos])

#Separar X(Características) y y (Objetivo)
X = df.drop('income', axis=1)

y = df['income']

print("\n--- Separación de datos ---")
print(f"Características (X): {X.shape[1]} columnas")
print(f"Objetivo (y): {y.name}")
#Creamos el codificador para el target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Verificamos la transformación:
# Muestra las clases que aprendio(0 <=50k y 1>50)
print(f"Clases detectadas en el taget {le.classes_}")

#Identificamos cuáles son las columnas de categoricas
columnas_categoricas = X.select_dtypes(include=['object']).columns

# Configuramos el Transformador de Columnas
# 'cat': nombre del paso
# OneHotEncoder: la herramienta
# columnas_categoricas: a qué columnas aplicarlo
# remainder='passthrough': las columnas que son números déjalas igual
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), columnas_categoricas)
    ],
    remainder='passthrough'
)

#Aplicamos la transformación 
X_encoded = preprocessor.fit_transform(X)

print("\n---Encodig finalizado--")
print(f"Nuevas dimensiones de X: {X_encoded.shape}")