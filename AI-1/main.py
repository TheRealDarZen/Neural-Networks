import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
# print(heart_disease.metadata)

# variable information
# print(heart_disease.variables)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

PATH_FOR_RESULTS = 'data'
DATA_TEXT_FILE_NAME = 'heart_disease_dataset_analysis_data.txt'

with open(f'{PATH_FOR_RESULTS}/{DATA_TEXT_FILE_NAME}', 'w', encoding='utf-8') as file:
    print(X.describe(), file=file)

for column in X:
    print(X[column].value_counts())
    with open(f'{PATH_FOR_RESULTS}/{DATA_TEXT_FILE_NAME}', 'a', encoding='utf-8') as file:
         print(X[column].value_counts(normalize=True), file=file)

    _, aX = plt.subplots()
    X[column].plot(kind='kde', color='black', ax=aX, secondary_y=True)
    X[column].plot(kind='hist', color='blue', ax=aX, bins=100)
    plt.title(f'{column} distribution')
    plt.savefig(f'{PATH_FOR_RESULTS}/{column}-distribution')
    plt.close()

    stat, p = shapiro(X[column])
    normal_dist = p > 0.05
    with open(f'{PATH_FOR_RESULTS}/{DATA_TEXT_FILE_NAME}', 'a', encoding='utf-8') as file:
        print(f'{column} - {normal_dist}', file=file)
