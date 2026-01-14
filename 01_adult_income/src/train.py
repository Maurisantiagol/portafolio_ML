import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Importamos la función que creamos en el otro archivo
from transformers import get_preprocessor

# --- 1. CONFIGURACIÓN DE RUTAS Y CARGA ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_data = os.path.join(BASE_DIR, 'data', 'adult', 'adult.data')

nombres = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
           'capital-loss', 'hours-per-week', 'native-country', 'income']

# Cargamos el archivo tratando el '?' como nulo
df = pd.read_csv(ruta_data, names=nombres, na_values='?', skipinitialspace=True)

# --- 2. PREPARACIÓN DE VARIABLES ---
X = df.drop('income', axis=1)
# Convertimos el target (income) a 0 y 1
le = LabelEncoder()
y = le.fit_transform(df['income'])

# Identificamos automáticamente qué columnas son números y cuáles texto
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# --- 3. DIVISIÓN TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. APLICAR TRANSFORMACIONES ---
preprocessor = get_preprocessor(num_cols, cat_cols)

# El entrenamiento solo "aprende" de X_train
X_train_final = preprocessor.fit_transform(X_train)
# El test solo se "transforma" con lo aprendido
X_test_final = preprocessor.transform(X_test)

# --- 5. ENTRENAMIENTO DEL MODELO ---
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train_final, y_train)

# --- 6. EVALUACIÓN ---
y_pred = modelo.predict(X_test_final)

print("\n" + "="*30)
print(" RESULTADOS DEL MODELO ")
print("="*30)
print(classification_report(y_test, y_pred))

# Mostrar Matriz de Confusión
print("\nMostrando Matriz de Confusión...")
ConfusionMatrixDisplay.from_estimator(modelo, X_test_final, y_test, 
                                      display_labels=le.classes_, 
                                      cmap='Blues')
plt.title("Matriz de Confusión - Regresión Logística")
plt.show()