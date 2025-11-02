import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def get_data():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Preprocessing
    X = X.dropna()
    y = y.loc[X.index]

    cat_var_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    X = pd.get_dummies(X, columns=cat_var_cols, drop_first=True, dtype=int)

    # y.loc[:, "num"] = y["num"].apply(lambda x: 1 if x != 0 else 0)

    X = (X - X.mean()) / X.std()

    X = X.to_numpy().T  # (features, samples)
    y = y.to_numpy().T  # (1, samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )

    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    return X_train, X_test, y_train, y_test

