from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.lang import builder
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from Function import pokemon_battle, predict, get_train
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock

import pandas as pd
import joblib



model = joblib.load("RandomForestClassifier.joblib")
poke_data=pd.read_csv('pokemon.csv')
pokemon_name = poke_data['Name'].values
combat_data=pd.read_csv('combats.csv')
train = get_train(poke_data,combat_data)


class MyGird(GridLayout):
    def __init__(self,**kwargs):
        Window.clearcolor = (1, 1, 1, 1)
        super(MyGird,self).__init__(**kwargs)
        self.cols = 1
        self.inside = GridLayout()
        self.inside.cols = 2



        self.inside.pokemon1_img = Image(source='pokemon_images\gif.gif')
        self.inside.pokemon1_img.size_hint_x = 1
        self.inside.pokemon1_img.size_hint_y = 1

        self.inside.pokemon2_img = Image(source='pokemon_images\gif.gif')
        self.inside.pokemon2_img.size_hint_x = 1
        self.inside.pokemon2_img.size_hint_y = 1

        self.inside.pokemon1 = TextInput(multiline = False)
        self.inside.pokemon2 = TextInput(multiline = False)
        self.inside.select1 = Button(text="SELECT FIRST POKEMON",on_press=self.select1)
        self.inside.select2 = Button(text="SELECT SECOND POKEMON",on_press=self.select2)


        self.add_widget(self.inside)
        self.button = Button(text="START BATTLE",font_size =40,on_press=self.pred)
        self.add_widget(self.button)


        # Adding widgets
        self.inside.add_widget(Label(text="NAME YOUR FIRST POKEMON",color=[0, 0, 0, 1]))
        self.inside.add_widget(Label(text="NAME YOUR SECOND POKEMON",color=[0, 0, 0, 1]))
        self.inside.add_widget(self.inside.pokemon1)
        self.inside.add_widget(self.inside.pokemon2)
        self.inside.add_widget(self.inside.select1)
        self.inside.add_widget(self.inside.select2)
        self.inside.add_widget(self.inside.pokemon1_img)
        self.inside.add_widget(self.inside.pokemon2_img)
        # self.inside.add_widget(self.inside.winer_img)


    def pred(self,instance):
        poke1 = self.inside.pokemon1.text
        poke2 = self.inside.pokemon2.text
        poke_battle,poke_name = pokemon_battle(poke1,poke2,poke_data,train)
        predict(poke_battle,poke_name,model)

    def select1(self,instance):
        poke1 = self.inside.pokemon1.text
        self.inside.pokemon1_img.source='pokemon_images\{0}.jpg'.format(poke1)
        self.inside.pokemon1_img.size_hint_x = 1
        self.inside.pokemon1_img.size_hint_y = 1


    def select2(self,instance):
        poke2 = self.inside.pokemon2.text
        self.inside.pokemon2_img.source='pokemon_images\{0}.jpg'.format(poke2)
        self.inside.pokemon2_img.size_hint_x = 1
        self.inside.pokemon2_img.size_hint_y = 1

    def my_callback(self,dt):
        pass

class Pokemon(App):
    def build(self):
        return MyGird()

if __name__ == "__main__":
    Pokemon().run()