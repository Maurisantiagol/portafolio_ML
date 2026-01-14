import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from transformers import get_medical_preprocessor

def ejecutar_experimento(dataframe, lista_num, lista_cat, nombre_prueba):
    """
    Función para entrenar y evaluar el modelo con diferentes configuraciones.
    """

    X = dataframe[lista_num + lista_cat]
    y = dataframe['charges']

    prepro = get_medical_preprocessor(lista_num, lista_cat)

    pipeline = Pipeline(steps=[
        ('preprocesamiento', prepro),
        ('modelo', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n>>> RESULTADOS: {nombre_prueba}")
    print(f"MAE (Error en $): ${mae:.2f}")
    print(f"R2 (Precisión): {r2:.4f}")
    print("-" * 30)

    return pipeline

# --- FLUJO PRINCIPAL ---

# Configuración de rutas
ruta_principal = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_csv = os.path.join(ruta_principal, 'data', 'insurance.csv')

# Carga de datos
if os.path.exists(ruta_csv):
    df = pd.read_csv(ruta_csv)

    # --- EXPERIMENTO 1: Base (Columnas originales) ---
    cols_num_base = ['age', 'bmi', 'children']
    cols_cat_base = ['sex', 'smoker', 'region']
    ejecutar_experimento(df, cols_num_base, cols_cat_base, "Modelo Base")

    # --- EXPERIMENTO 2: Ingeniería de Características ---
    # Creamos copias para no afectar el dataframe original
    df_mejorado = df.copy()
    
    # Teoría A: La edad no es lineal (Polinomio)
    df_mejorado['age_2'] = df_mejorado['age'] ** 2
    
    # Teoría B: Interacción Obesidad + Fumador (El factor de riesgo más alto)
    # Convertimos 'smoker' a 1 y 0 para multiplicar
    df_mejorado['bmi_smoker'] = df_mejorado['bmi'] * df_mejorado['smoker'].map({'yes': 1, 'no': 0})
    
    # Nuevas listas de columnas incluyendo las creadas
    cols_num_mejorado = ['age', 'bmi', 'children', 'age_2', 'bmi_smoker']
    
    ejecutar_experimento(df_mejorado, cols_num_mejorado, cols_cat_base, "Modelo con Ingeniería (Edad^2 + Interacción)")

else:
    print(f"Error: No se encontró el archivo en {ruta_csv}")