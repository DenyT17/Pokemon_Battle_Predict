# Pokemon Battle Prediction 🐲

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
|Analysis percent of wins battle of pokemon from each generation| <img src="https://user-images.githubusercontent.com/122997699/223529358-ce338367-2023-484f-9eb0-46abc76258fb.png" width="450" height="250">|Bar graph, showing percent of wins to pokemon from each generations. According to the chart in second and third generation pokemons are the powerful. The most weakest pokemons are in fourth generation.|
|Top 10 pokemons with most number of wins|<img src="https://user-images.githubusercontent.com/122997699/223529238-67eb0b0f-c470-4f40-90df-c49f82f55357.png" width="450" height="250">|This graph showing ten pokemon with the most wins, and this is Mewtwo |
|Top 10 pokemons with the best win ratio|<img src="https://user-images.githubusercontent.com/122997699/223529149-a0316617-9a2e-4c40-9419-e3a3f4ec27ea.png" width="450" height="250">| This graph showing ten pokemon with the best win ratio, and this is Mega Areodactyl|


#### Pokemon type encoding and checking the effect.

Every pokemon has a type. Pokemon Type may be important, because for example water pokemon have higher chance to win against fire pokemon than against rock pokemon.
So I decide to take into consideration pokemon first type. For these reason I must encode pokemon type in to number. I will use two ways to do this and compare the results.

* Using pandas ***get_dummies*** method
Result of using this method is creation of new columns, one for every type in first and second pokemon. 
In the column that describes the type of a given pokemon appears 1 and in the others 0. Example result:

![image](https://user-images.githubusercontent.com/122997699/223490284-acc765e0-5d07-4d7d-8e4d-c47002b89b95.png)

![image](https://user-images.githubusercontent.com/122997699/223488670-771834cd-b661-4d35-9a71-ce7e5ca18c6e.png)

* Using sklearn ***LabelEncoder*** method
Result of using this method is changing every pokemon type to the corresponding number. Example result:Example result:

![image](https://user-images.githubusercontent.com/122997699/223487297-4e275efe-962d-470f-bcdd-b94ce6b0ec18.png)

![image](https://user-images.githubusercontent.com/122997699/223487340-40f92b1e-4633-4011-b67b-61b043f51c14.png)

Second method giving better accuracy, so i choose it. 
 
### Rescaling data and choosing classifier with the highest accuracy.
#### Rescaling data
In this part of project first i rescale dataset. Thanks to which accuracy may be higher and training model will be faster.
```python
for column in x_train.columns:
    ages_data = np.array(train[column]).reshape(-1, 1)
    x_train[column] = StandardScaler().fit_transform(ages_data)
```
***ScandarScaler()*** function standardizes features by removing the mean and scaling to unit variance.

The part of code suscped for rescaling and encoding data is in ***enc_resc*** function. 

#### Choosing classifier with the highest accuracy.

In this case I create ***choose classifier*** function. This function return table with names and accuracy of chosen classifiers. 
To finally select the classifier, cross-validation will be used. The part of the code responsible for this operation is in ***choose_classifier*** function.

```python
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
```

Results :

![image](https://user-images.githubusercontent.com/122997699/223502637-934665ee-e000-4de5-8972-41e7c661dba6.png)

As you can see, Random Forest Classifier has the best accuracy from chosen models. Thanks to this information, I can use this model for next steps in my project. 

The trained model is in the files under the name: _RandomForestClassifier.joblib_

## Downloading pokemon pictures using web scraping

Because in my GUI i want to use images showing selected pokemon i must gain pokemons images. 
For this, I create a new file  ***img_download.py***. I will fownload images from [this page](https://pokemondb.net/pokedex/all))
Thanks to this I have folder with pokemon images, które posiadają nazwy kompatybilne z tymi w pliku pokemon.csv.


I manually downloaded pictures for the following pokemons:
-Farfetch'd, 
-Mr. Mime, 
-DeoxysAttack Forme, 
-Mime Jr., 
-Giratina Altered Forme, 
-Giratina Origin Forme, 
-Shaymin Land Forme, 
-Shaymin Sky Forme, 
-Flabébé


## Create GUI in Kivy
## Next goals 🏆⌛
* Create GUI


