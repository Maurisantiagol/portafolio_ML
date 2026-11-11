# portafolio_ML

Colección de experimentos de Aprendizaje Supervisado utilizando Scikit-Learn y Python 3.10. Incluye proyectos de clasificación y regresión desarrollados en Jupyter Notebooks con Pandas, NumPy, Matplotlib y Seaborn.

## Objetivo

Practicar el flujo completo de un proyecto de aprendizaje supervisado: limpieza de datos, preprocesamiento, entrenamiento, evaluación y visualización de resultados, con dos conjuntos de datos clásicos.

## Proyectos

### 01_adult_income — Clasificación de ingresos

Predice si una persona gana más de 50 mil dólares al año a partir de datos del censo.

- **Datos:** `data/adult/adult.data`, con 15 columnas: `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country` e `income` (la variable objetivo). El valor `?` se trata como nulo.
- **Modelo:** `LogisticRegression(max_iter=1000)` dentro de un `Pipeline` de scikit-learn.
- **Preprocesamiento** (`transformers.py`): variables numéricas con imputación por la mediana y `StandardScaler`; variables categóricas con imputación por el valor más frecuente y `OneHotEncoder(handle_unknown='ignore')`. La variable `income` se codifica con `LabelEncoder`.
- **Evaluación:** división entrenamiento/prueba 80/20 (`random_state=42`); se imprime el `classification_report` y se muestra la matriz de confusión (`ConfusionMatrixDisplay`, `cmap='Blues'`).
- **Archivos:** `src/cleaner.py`, `src/train.py` y `src/transformers.py`.

### 02_medical_costs — Regresión de costos médicos

Predice el costo médico anual (`charges`) de una persona a partir de sus características.

- **Datos:** `data/insurance.csv`.
- **Modelo:** `LinearRegression` dentro de un `Pipeline` de scikit-learn.
- **Experimento 1 (base):** variables numéricas `age`, `bmi` y `children`; variables categóricas `sex`, `smoker` y `region`.
- **Experimento 2 (ingeniería de características):** además de las variables base, se agregan `age_2 = age ** 2` y `bmi_smoker = bmi * smoker` (con `smoker` como 1 = sí, 0 = no).
- **Evaluación:** división entrenamiento/prueba 70/30 (`random_state=42`); se imprimen el MAE (en dólares) y el R².
- **Archivos:** `src/train.py` y `src/transformers.py`.

> **Nota sobre los resultados:** los scripts imprimen las métricas al ejecutarse, pero en este repositorio no hay valores de resultados guardados. Para verlos, corre los scripts como se indica abajo.

## Instalación y ejecución

Requisitos: Python 3.10.

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

El repositorio no incluye `requirements.txt`; las dependencias son las que usan los scripts (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`).

Para correr cada experimento desde la carpeta del proyecto:

```bash
cd 01_adult_income
python src/train.py

cd ../02_medical_costs
python src/train.py
```

## Estructura

```
portafolio_ML/
├── 01_adult_income/
│   ├── data/
│   └── src/
│       ├── cleaner.py
│       ├── train.py
│       └── transformers.py
├── 02_medical_costs/
│   ├── data/
│   └── src/
│       ├── train.py
│       └── transformers.py
├── .gitignore
└── README.md
```

## Qué le falta / pendientes

- Un `requirements.txt` con las versiones exactas de las dependencias.
- Guardar los resultados (métricas y gráficas) de cada experimento en el repositorio.
- Los notebooks `.ipynb` que menciona la descripción, si existen.

## Licencia

MIT. Ver [LICENSE](LICENSE).
