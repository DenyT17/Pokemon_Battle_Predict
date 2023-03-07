import pandas as pd
import numpy as np
from Function import analysis, pokemon_battle, choose_classifier
from sklearn import (model_selection as skms)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import  cross_val_score

# Read data from csv files
pd.set_option('display.max_columns', None)
poke_data=pd.read_csv('pokemon.csv')
combat_data=pd.read_csv('combats.csv')
test_data=pd.read_csv('tests.csv')

# Data Analisis and Preparation
train, y_train = analysis(poke_data,combat_data)
x_train = train.drop(
    ["Winner", 'Type 1', 'Type 2', 'Oponent Type 1', 'Oponent Type 2', 'Legendary', 'Oponent Legendary'], axis=1)

# Data Rescaling
for column in x_train.columns:
    ages_data = np.array(train[column]).reshape(-1, 1)
    x_train[column] = StandardScaler().fit_transform(ages_data)
(poke_train_ftrs, poke_test_ftrs,
 poke_train_trg, poke_test_trg,) = skms.train_test_split(x_train, y_train, test_size=.3)
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    SGDClassifier(max_iter=1000, tol=0.01),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    RandomForestClassifier(),
]

# Choosing classifier
choose_classifier(classifiers,poke_train_ftrs,poke_train_trg,poke_test_ftrs,poke_test_trg)

# Cross validation
scores = cross_val_score(RandomForestClassifier(),poke_train_ftrs, poke_train_trg, cv=5)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Training model
model = RandomForestClassifier().fit(poke_train_ftrs, poke_train_trg)
preds = model.predict(poke_test_ftrs)
filename = "RandomForestClassifier.joblib"
joblib.dump(model, filename)

# Creating combat data
pokemon_battle_input = pokemon_battle('Virizion','Terrakion',poke_data,x_train)
model = joblib.load("RandomForestClassifier.joblib")

