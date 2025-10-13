import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

PATH_FOR_RESULTS = 'data'
DATA_TEXT_FILE_NAME = 'matrix.csv'

cat_var_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

matrix = pd.get_dummies(X, columns=cat_var_cols, drop_first=True, dtype=int)

matrix.to_csv(f'{PATH_FOR_RESULTS}/{DATA_TEXT_FILE_NAME}', index=False)
