from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from Function import pokemon_battle, predict, get_train
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.popup import Popup
import pandas as pd
import joblib


model = joblib.load("RandomForestClassifier.joblib")
poke_data=pd.read_csv('pokemon.csv')
poke_data = poke_data.dropna(subset=['Name'])
pokemon_name = list(poke_data['Name'].values)
combat_data=pd.read_csv('combats.csv')
train = get_train(poke_data,combat_data)


class MyGird(GridLayout):
    def __init__(self,**kwargs):
        Window.clearcolor = (1, 1, 1, 1)
        super(MyGird,self).__init__(**kwargs)
        self.cols = 1
        self.inside = GridLayout()
        self.inside.cols = 2
        self.image = GridLayout()
        self.image.cols = 2
        self.image.pokemon1_img = Image(source='pokemon_images\gif.gif')
        self.image.pokemon1_img.size_hint_x = 1
        self.image.pokemon1_img.size_hint_y = 1
        self.image.pokemon2_img = Image(source='pokemon_images\gif.gif')
        self.image.pokemon2_img.size_hint_x = 1
        self.image.pokemon2_img.size_hint_y = 1
        self.winer_img = Image(source='pokemon_images\winer.jpg')
        self.winer_img.size_hint_x = 3
        self.winer_img.size_hint_y = 3
        self.inside.pokemon1 = TextInput(multiline = False)
        self.inside.pokemon2 = TextInput(multiline = False)
        self.inside.select1 = Button(text="SELECT FIRST POKEMON",on_press=self.select1)
        self.inside.select2 = Button(text="SELECT SECOND POKEMON",on_press=self.select2)
        self.add_widget(self.inside)
        self.add_widget(self.image)
        self.button = Button(text="START BATTLE",font_size =40,on_press=self.pred)

        # Adding widgets
        self.inside.add_widget(Label(text="NAME YOUR FIRST POKEMON",color=[0, 0, 0, 1]))
        self.inside.add_widget(Label(text="NAME YOUR SECOND POKEMON",color=[0, 0, 0, 1]))
        self.inside.add_widget(self.inside.pokemon1)
        self.inside.add_widget(self.inside.pokemon2)
        self.inside.add_widget(self.inside.select1)
        self.inside.add_widget(self.inside.select2)
        self.image.add_widget(self.image.pokemon1_img)
        self.image.add_widget(self.image.pokemon2_img)
        self.add_widget(self.button)
        self.add_widget(Label(text="WINNER IS :", color=[0, 0, 0, 1]))
        self.add_widget(self.winer_img)

    def pred(self,instance):
        poke1 = self.inside.pokemon1.text
        poke2 = self.inside.pokemon2.text
        poke_battle,poke_name = pokemon_battle(poke1,poke2,poke_data,train)
        if predict(poke_battle,poke_name,model) == 0:
            self.winer_img.source=self.pokemon_1_source
        elif predict(poke_battle,poke_name,model) == 1:
            self.winer_img.source=self.pokemon_2_source
    def select1(self,instance):
        poke1 = self.inside.pokemon1.text
        if poke1 in pokemon_name:
            self.pokemon_1_source = 'pokemon_images\{0}.jpg'.format(poke1)
            self.image.pokemon1_img.source='pokemon_images\{0}.jpg'.format(poke1)
            self.image.pokemon1_img.size_hint_x = 1
            self.image.pokemon1_img.size_hint_y = 1
        else:
            self.wrong_name()
    def select2(self,instance):
        poke2 = self.inside.pokemon2.text
        if poke2 in pokemon_name:
            self.pokemon_2_source = 'pokemon_images\{0}.jpg'.format(poke2)
            self.image.pokemon2_img.source='pokemon_images\{0}.jpg'.format(poke2)
            self.image.pokemon2_img.size_hint_x = 1
            self.image.pokemon2_img.size_hint_y = 1
        else:
            self.wrong_name()
    def my_callback(self,dt):
        pass
    def wrong_name(self):
        layout = GridLayout(cols=1, padding=10)
        popupLabel = Label(text="POKEMON NAME IS WRONG\nPLEAS TRY AGAIN")
        closeButton = Button(text="Close")
        layout.add_widget(popupLabel)
        layout.add_widget(closeButton)
        popup = Popup(title='Demo Popup',
                      content=layout,
                      size_hint=(None, None), size=(250, 200))
        popup.open()
        closeButton.bind(on_press=popup.dismiss)
class Pokemon(App):
    def build(self):
        return MyGird()

if __name__ == "__main__":
    Pokemon().run()