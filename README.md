# Pokemon Analysis 🐲

## Technologies 💡
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Sklearn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Description 
In this project i analyze and prepare pokemon dataset.I find pokemon with the best win ratio and with most number of wins.
I train model with the best with the greatest possible accuracy.At the end I make simple GUI to visualisation pokemon battle winner prediction.
I will reguraly upadte code and description. 

## Project stages

* Analisis pokemon dataset 

* Rescaling data

* Creating model with highest accuracy as possible

* Creating basic API to simulation pokemon battle

## Dataset📁
Dataset used in this project you can find below this [link](https://www.kaggle.com/datasets/terminus7/pokemon-challenge)

## Project implementation description 🔍
### Analysis
In this part, I analisis basic pokemon stats and put in to graph. Part of cod responsible is in ***analysis*** function. 
In addition, in this part, I fill in all the Nan values, and create a new DataFrames that contains the required information to train the model, such as:

* statistics of both pokemons (from the pokemon csv file) participating in the battle (from the combats.csv file),
* winner of every battle

This DataFrame is use to train model.

#### Analysis Graphs:

**To zoom each graph click on it.**

| Name| Graph | Short Description |
| -- | ------------- | ------------- |
| Number of pokemon in each type|<img src="https://user-images.githubusercontent.com/122997699/216782334-e50775c8-0c3f-4a5b-af42-a24ce95c1184.png" width="600" height="250">| Bar chart showing number of pokemon in each type.There are the most water-type and normal-type Pokemon, and the least flying-type.  |
|Percent of number of pokemon in each generation.| <img src="https://user-images.githubusercontent.com/122997699/216782333-613f6a6a-d782-4b54-9f63-15ce4e5e73c4.png" width="600" height="250">|A pie chart showing how much of the Pokemon set are Pokemon from a each generation.  The fewest Pokemon come from the sixth generation. The number of Pokemon from other generations is similar.|
|Analysis percent of wins battle of pokemon from each generation| <img src="https://user-images.githubusercontent.com/122997699/216782337-903efa19-b692-46f4-adf1-ce39fd7fa274.png" width="450" height="250">|Bar graph, showing percent of wins to pokemon from each generations. According to the chart in second and third generation pokemons are the weakest. The most powerful pokemons are in fourth generation.|
|Top 10 pokemons with most number of wins|<img src="https://user-images.githubusercontent.com/122997699/216782341-8a87e229-0719-4a7b-a7c0-a2b4701ef3b6.png" width="450" height="250">|This graph showing ten pokemon with the most wins. |
|Top 10 pokemons with the best win ratio|<img src="https://user-images.githubusercontent.com/122997699/216782343-806d6b73-11da-4636-b1e3-d68e6ee8b09c.png" width="450" height="250">| This graph showing ten pokemon with the best win ratio|

### Rescaling data and choosing classifier with the highest accuracy.
#### Rescaling data
In this part of project first i rescale dataset. Thanks to which accuracy may be higher and training model will be faster.
```python
for column in x_train.columns:
    ages_data = np.array(train[column]).reshape(-1, 1)
    x_train[column] = StandardScaler().fit_transform(ages_data)
```
***ScandarScaler()*** function standardizes features by removing the mean and scaling to unit variance.
#### Choosing classifier with the highest accuracy.

In this case I create ***choose classifier*** function. This function return table with names and accuracy of chosen classifiers. 
```python
def choose_classifier(classifiers,poke_train_ftrs,poke_train_trg,poke_test_ftrs,poke_test_trg):
    rank = pd.DataFrame(columns=["Name","Accuracy"])
    i=0
    for classifier in classifiers:
        fit = classifier.fit(poke_train_ftrs, poke_train_trg)
        preds = fit.predict(poke_test_ftrs)
        name = classifier.__class__.__name__
        rank.loc[i,"Name"] = name
        rank.loc[i,"Accuracy"] = metrics.accuracy_score(poke_test_trg, preds)
        i+=1
    rank = rank.sort_values(by="Accuracy",ascending=False).reset_index(drop=True)
    print(tabulate(rank, headers = 'keys', tablefmt = "rounded_outline"))
```
![image](https://user-images.githubusercontent.com/122997699/223136129-e5d49086-6a5c-444b-887f-050a1c6031aa.png)


According this information I choose _RandomForestClassifier_.

## Next goals 🏆⌛
* Create GUI
* Try different classifiers and choose the best one
* Cross validation and Feature enginering to increase accuracy 
