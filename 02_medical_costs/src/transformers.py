from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
def get_medical_preprocessor(cols_num,cols_cat):
    numeric_transformer = Pipeline(steps=[
        ('imputador', SimpleImputer(strategy='mean')),
        ('escalador', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputador',SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numericos', numeric_transformer, cols_num),
            ('categoricos', categorical_transformer, cols_cat)
        ]
    )
    return preprocessor