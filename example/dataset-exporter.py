import pandas as pd 
import numpy as np 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

random_state = 2


def dataPreprocessing():
    # Loading dataset
    penguins_data = pd.read_csv('penguins.csv')
    # Preprocess
    penguins_data.dropna(inplace=True)
    penguins_data.drop(penguins_data.index[penguins_data['sex'] == '.'].tolist(),inplace=True)
    penguins_data['species']=penguins_data['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})
    penguins_data.reset_index(inplace=True)
    penguins_data.drop('index', axis=1, inplace=True)
    df_male = pd.get_dummies(data=penguins_data['sex'], drop_first=True)
    penguins_data = pd.concat([penguins_data,df_male], axis=1)
    df_island = pd.get_dummies(data=penguins_data['island'])
    penguins_data = pd.concat([penguins_data,df_island], axis=1)
    pgn = penguins_data.drop(['island','sex'], axis=1)
    df_island.head()
    X = pgn.drop('species', axis=1)
    y = pgn['species']

    # Normalization
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X = np.ascontiguousarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=101)
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=.7, random_state=random_state)
X_train, X_test, y_train, y_test = dataPreprocessing()
X_train.rename(columns={'MALE': 'is_male', 'Biscoe': 'island_biscoe', 'Dream': 'island_dream', 'Torgersen': 'island_torgersen'}, inplace=True)
print(X_train.head())
# Export training dataset into csv files
# pd.DataFrame(X_train).to_csv("train_x_iris_unscaled.csv", index=False)
# pd.DataFrame(y_train).to_csv("train_y_iris_unscaled.csv", index=False)
pd.DataFrame(X_train).to_csv("train_x_penguins_unscaled.csv", index=False)
pd.DataFrame(y_train).to_csv("train_y_penguins_unscaled.csv", index=False)