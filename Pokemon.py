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

# Read data from csv files
pd.set_option('display.max_columns', None)
poke_data=pd.read_csv('pokemon.csv')
combat_data=pd.read_csv('combats.csv')
test_data=pd.read_csv('tests.csv')

train, y_train = analisis(poke_data,combat_data)

# Winner prediction
x_train = train.drop(
    ["Winner", 'Type 1', 'Type 2', 'Oponent Type 1', 'Oponent Type 2', 'Legendary', 'Oponent Legendary'], axis=1)
(poke_train_ftrrs, poke_test_ftrs,
 poke_train_trg, poke_test_trg,) = skms.train_test_split(x_train, y_train, test_size=.3)
model = DecisionTreeClassifier()

model = RandomForestClassifier()
fit = model.fit(poke_train_ftrrs, poke_train_trg)
preds = fit.predict(poke_test_ftrs)
print("Fit: ", metrics.accuracy_score(poke_test_trg, preds))