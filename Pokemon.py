import pandas as pd
import numpy as np
from Analisis import analisis
from sklearn import (datasets, metrics,model_selection as skms,neighbors)
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Read data from csv files
pd.set_option('display.max_columns', None)
poke_data=pd.read_csv('pokemon.csv')
combat_data=pd.read_csv('combats.csv')
test_data=pd.read_csv('tests.csv')

# Data Analisis and Preparation
train, y_train = analisis(poke_data,combat_data)
x_train = train.drop(
    ["Winner", 'Type 1', 'Type 2', 'Oponent Type 1', 'Oponent Type 2', 'Legendary', 'Oponent Legendary'], axis=1)
# Creating combat data

name1 = "Bulbasaur"
name2 = "Ivysaur"

pokemon1 = poke_data.loc[poke_data['Name'] == name1].reset_index()
pokemon2 = poke_data.loc[poke_data['Name'] == name2].reset_index()
pokemon1 = pokemon1.drop(['index','#','Name','Type 1','Type 2','Legendary'],axis=1)
pokemon2 = pokemon2.drop(['index','#','Name','Type 1','Type 2','Legendary'],axis=1)
pokemon2.rename(columns={ 'Type 1': 'Oponent Type 1', 'Type 2': 'Oponent Type 2',
                               'HP': 'Oponent HP', 'Attack': 'Oponent Attack', 'Defense': 'Oponent Defense',
                               'Sp. Atk': 'Oponent Sp. Atk', 'Sp. Def': 'Oponent Sp. Def', 'Speed': 'Oponent Speed',
                               'Generation': 'Oponent Generation'
                               }, inplace=True)

pokemon_battle = pd.concat([pokemon1,pokemon2],axis=1)
pokemon_battle = pd.concat([pokemon_battle,x_train],axis=0).reset_index()
pokemon_battle = pokemon_battle.drop(['index'],axis=1)
print(pokemon_battle)
for column in pokemon_battle.columns:
    ages_data = np.array(train[column]).reshape(-1, 1)
    pokemon_battle[column] = StandardScaler().fit_transform(ages_data)










# Winner prediction
x_train = train.drop(
    ["Winner", 'Type 1', 'Type 2', 'Oponent Type 1', 'Oponent Type 2', 'Legendary', 'Oponent Legendary'], axis=1)

# Data Rescaling
for column in x_train.columns:
    ages_data = np.array(train[column]).reshape(-1, 1)
    x_train[column] = StandardScaler().fit_transform(ages_data)
(poke_train_ftrrs, poke_test_ftrs,
 poke_train_trg, poke_test_trg,) = skms.train_test_split(x_train, y_train, test_size=.3)

model =RandomForestClassifier()
fit = model.fit(poke_train_ftrrs, poke_train_trg)
preds = fit.predict(poke_test_ftrs)
print("Fit: ", metrics.accuracy_score(poke_test_trg, preds))