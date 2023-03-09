import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn import (datasets, metrics,model_selection as skms,naive_bayes,neighbors)
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# A function that calculates what percentage of the data set is a given generation
def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

def analysis(poke_data,combat_data):

    # Fill NaN values
    poke_data["Name"].fillna(value="Unknown", inplace=True)
    plt.style.use('seaborn-v0_8-pastel')
    # Pokemon generation
    generation_number = poke_data["Generation"].value_counts()
    generation = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]
    counts = generation_number.values
    ax, fig = plt.subplots(figsize=(10, 7))
    plt.pie(counts, labels=generation,
            autopct=lambda pct: func(pct, counts))
    plt.legend(loc='lower left', bbox_to_anchor=(-0.4, 0, 0, 0))
    plt.title('Pokemon generation', fontsize=20)

    # Main pokemon type
    types_number = poke_data["Type 1"].value_counts()
    types = types_number.index
    counts_types = types_number.values
    ax, fig = plt.subplots(figsize=(10, 7))
    plt.bar(x=types, height=counts_types, )
    plt.xticks(rotation=30, ha='right')
    plt.title('Main pokemons type', fontsize=20)
    plt.grid()


    combat_data.rename(columns={"First_pokemon": "#"}, inplace=True)
    own_stats = combat_data.merge(poke_data, on='#', how='left')
    second_pokemon = poke_data.rename(columns={"#": "Second_pokemon"})
    second_pokemon.rename(columns={'Name': 'Oponent Name', 'Type 1': 'Oponent Type 1', 'Type 2': 'Oponent Type 2',
                                   'HP': 'Oponent HP', 'Attack': 'Oponent Attack', 'Defense': 'Oponent Defense',
                                   'Sp. Atk': 'Oponent Sp. Atk', 'Sp. Def': 'Oponent Sp. Def', 'Speed': 'Oponent Speed',
                                   'Generation': 'Oponent Generation', 'Legendary': 'Oponent Legendary'
                                   }, inplace=True)
    train = own_stats.merge(second_pokemon, on="Second_pokemon", how='left')
    train.rename(columns={'#': 'First_pokemon'}, inplace=True)
    train["Winner"] = np.where(train["Winner"] == train['First_pokemon'], 0, 1)
    train = train.drop(['First_pokemon', 'Second_pokemon', 'Name', 'Oponent Name'], axis=1)
    train = train.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna('NA'))
    ax, fig = plt.subplots(figsize=(10, 7))
    sns.barplot(y='Winner', x='Generation', data=train)
    y_train = train['Winner']
    plt.title('Percentage of wins battle for each generation of pokemon', fontsize=20)
    plt.grid()

    # Pokemon with most wins
    pokemon_battle = combat_data.rename(columns={'#': 'First_pokemon', 'Winner': '#'})
    pokemon_battle = pokemon_battle.merge(poke_data[['#', 'Name']], on='#')
    pokemon_battle.drop('#', axis='columns', inplace=True)
    pokemon_battle = pokemon_battle.rename(columns={'Name': 'Winner'})
    top10 = pokemon_battle['Winner'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.bar(x=top10.index, height=top10.values)
    plt.xticks(rotation=30)
    plt.ylabel("Winns")
    plt.xlabel('Pokemon name')
    plt.title('Pokemon with the most wins', fontsize=15)
    plt.tight_layout()
    for i in range(len(top10.index)):
        plt.text(i, top10.values[i], top10.values[i], horizontalalignment='center')

    # Pokemon with most win ration
    pokemon_name = poke_data['Name'].values
    pokemon_battle = pokemon_battle.rename(columns={'First_pokemon': '#'})
    pokemon_battle = pokemon_battle.merge(poke_data[['#', 'Name']], on='#')
    pokemon_battle.drop('#', axis='columns', inplace=True)
    pokemon_battle = pokemon_battle.rename(columns={'Name': 'First_pokemon'})
    pokemon_battle = pokemon_battle.rename(columns={'Second_pokemon': '#'})
    pokemon_battle = pokemon_battle.merge(poke_data[['#', 'Name']], on='#')
    pokemon_battle.drop('#', axis='columns', inplace=True)
    pokemon_battle = pokemon_battle.rename(columns={'Name': 'Second_pokemon'})
    battles = (pokemon_battle['First_pokemon'].value_counts() + pokemon_battle[
        'Second_pokemon'].value_counts()).to_frame()
    wins = pokemon_battle['Winner'].value_counts().to_frame()
    battles = battles.reset_index()
    battles = battles.rename(columns={'index': 'Name', 0: 'Battles'})
    wins = wins.reset_index()
    wins = wins.rename(columns={'index': 'Name', 'Winner': 'Number of wins'})
    pokemon_win_ratio = battles.merge(wins[['Name', 'Number of wins']], on='Name')
    pokemon_win_ratio['Win ratio'] = (pokemon_win_ratio['Number of wins'] / pokemon_win_ratio['Battles']) * 100
    pokemon_win_ratio['Win ratio'] = round(pokemon_win_ratio['Win ratio'], 2)
    pokemon_win_ratio = pokemon_win_ratio.sort_values(by=['Win ratio'], ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    Name = pokemon_win_ratio['Name'].head(10)
    Values = pokemon_win_ratio['Win ratio'].head(10)
    plt.bar(x=Name, height=Values)
    plt.xticks(rotation=30)
    plt.ylabel("Win ratio")
    plt.xlabel('Pokemon name')
    plt.title('Pokemon with the best win raio', fontsize=15)
    for i in range(len(Name)):
        plt.text(i, Values[i], Values[i], horizontalalignment='center')
    plt.tight_layout()
    plt.show()

    return train,y_train
def enc_resc(x_train):
    features = ['Type 1', 'Oponent Type 1']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(x_train[feature])
        x_train[feature] = le.transform(x_train[feature])
    # Data Rescaling
    for column in x_train.columns:
        ages_data = np.array(x_train[column]).reshape(-1, 1)
        x_train[column] = StandardScaler().fit_transform(ages_data)
    return x_train
def choose_classifier(classifiers,poke_train_ftrs,poke_train_trg,poke_test_ftrs,poke_test_trg):
    rank = pd.DataFrame(columns=["Name","Accuracy","Average CV Score"])
    i=0
    for classifier in classifiers:
        fit = classifier.fit(poke_train_ftrs, poke_train_trg)
        preds = fit.predict(poke_test_ftrs)
        name = classifier.__class__.__name__
        scores = cross_val_score(classifier,poke_train_ftrs, poke_train_trg, cv=5)
        rank.loc[i,"Name"] = name
        rank.loc[i,"Accuracy"] = metrics.accuracy_score(poke_test_trg, preds)
        rank.loc[i,"Average CV Score"] = scores.mean()
        i+=1
    rank = rank.sort_values(by="Average CV Score",ascending=False).reset_index(drop=True)
    print(tabulate(rank, headers = 'keys', tablefmt = "rounded_outline"))

def pokemon_battle(pokemon1,pokemon2,poke_data,train):
    name1=pokemon1
    name2=pokemon2
    x_train = train.drop(
        ["Winner", 'Type 2', 'Oponent Type 2', 'Legendary', 'Oponent Legendary'], axis=1)
    pokemon1 = poke_data.loc[poke_data['Name'] == pokemon1].reset_index()
    pokemon2 = poke_data.loc[poke_data['Name'] == pokemon2].reset_index()
    pokemon1 = pokemon1.drop(['index','#','Name','Type 2','Legendary'],axis=1)
    pokemon2 = pokemon2.drop(['index','#','Name','Type 2','Legendary'],axis=1)
    pokemon2.rename(columns={ 'Type 1': 'Oponent Type 1', 'Type 2': 'Oponent Type 2',
                                   'HP': 'Oponent HP', 'Attack': 'Oponent Attack', 'Defense': 'Oponent Defense',
                                   'Sp. Atk': 'Oponent Sp. Atk', 'Sp. Def': 'Oponent Sp. Def', 'Speed': 'Oponent Speed',
                                   'Generation': 'Oponent Generation'
                                   }, inplace=True)

    pokemon_battle = pd.concat([pokemon1, pokemon2], axis=1)
    pokemon_battle = pd.concat([pokemon_battle, x_train], axis=0).reset_index(drop=True)
    pokemon_battle = enc_resc(pokemon_battle)

    return pokemon_battle.iloc[:1], [name1,name2]

def predict(pokemon_battle,pokemon_name,model):
    pred = model.predict(pokemon_battle)
    return pred

def get_train(poke_data,combat_data):
    poke_data["Name"].fillna(value="Unknown", inplace=True)
    combat_data.rename(columns={"First_pokemon": "#"}, inplace=True)
    own_stats = combat_data.merge(poke_data, on='#', how='left')
    second_pokemon = poke_data.rename(columns={"#": "Second_pokemon"})
    second_pokemon.rename(columns={'Name': 'Oponent Name', 'Type 1': 'Oponent Type 1', 'Type 2': 'Oponent Type 2',
                                   'HP': 'Oponent HP', 'Attack': 'Oponent Attack', 'Defense': 'Oponent Defense',
                                   'Sp. Atk': 'Oponent Sp. Atk', 'Sp. Def': 'Oponent Sp. Def', 'Speed': 'Oponent Speed',
                                   'Generation': 'Oponent Generation', 'Legendary': 'Oponent Legendary'
                                   }, inplace=True)
    train = own_stats.merge(second_pokemon, on="Second_pokemon", how='left')
    train.rename(columns={'#': 'First_pokemon'}, inplace=True)
    train["Winner"] = np.where(train["Winner"] == train['First_pokemon'], 0, 1)
    train = train.drop(['First_pokemon', 'Second_pokemon', 'Name', 'Oponent Name'], axis=1)
    train = train.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna('NA'))
    return train