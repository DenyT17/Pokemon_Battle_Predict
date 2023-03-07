import pandas as pd
import numpy as np
from Function import analysis, pokemon_battle, choose_classifier, enc_resc
from sklearn import (model_selection as skms)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import joblib
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
# Rescaling and encoding data
# x_train = train.drop(
#     ["Winner", 'Type 2', 'Oponent Type 2', 'Legendary', 'Oponent Legendary'], axis=1)
# x_train = enc_resc(x_train)

# Spliting data
# (poke_train_ftrs, poke_test_ftrs,
#  poke_train_trg, poke_test_trg,) = skms.train_test_split(x_train, y_train, test_size=.3)
# Defining classifiers
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
# choose_classifier(classifiers,poke_train_ftrs,poke_train_trg,poke_test_ftrs,poke_test_trg)


# Training model
# model = RandomForestClassifier().fit(poke_train_ftrs, poke_train_trg)
# preds = model.predict(poke_test_ftrs)
# filename = "RandomForestClassifier.joblib"
# joblib.dump(model, filename)
#
# Creating combat data
model = joblib.load("RandomForestClassifier.joblib")


pokemon_battle_input,pokemon_name = pokemon_battle('Larvitar','Nuzleaf',poke_data,train)
pred = model.predict(pokemon_battle_input)
print("Winner is {0} ".format(pokemon_name[pred[0]]))
print(model.predict_proba(pokemon_battle_input))

pokemon_battle_input,pokemon_name = pokemon_battle('Virizion','Terrakion',poke_data,train)
pred = model.predict(pokemon_battle_input)
print("Winner is {0} ".format(pokemon_name[pred[0]]))
print(model.predict_proba(pokemon_battle_input))

pokemon_battle_input,pokemon_name = pokemon_battle('Mega Venusaur','Bulbasaur',poke_data,train)
pred = model.predict(pokemon_battle_input)
print("Winner is {0} ".format(pokemon_name[pred[0]]))
print(model.predict_proba(pokemon_battle_input))

pokemon_battle_input,pokemon_name = pokemon_battle('Magikarp','Gyarados',poke_data,train)
pred = model.predict(pokemon_battle_input)
print("Winner is {0} ".format(pokemon_name[pred[0]]))
print(model.predict_proba(pokemon_battle_input))