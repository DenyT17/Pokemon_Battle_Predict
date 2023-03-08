import requests
import shutil
import pandas as pd

poke_data=pd.read_csv('pokemon.csv')
poke_data = poke_data.dropna(subset=['Name'])
pokemon_name = poke_data['Name'].values
error_name = []

def swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))

for orginal_name in pokemon_name:
    name = orginal_name.replace(' ', '-').lower()
    url = "https://img.pokemondb.net/artwork/large/{0}.jpg".format(name)
    file_name = 'pokemon_images/{0}.jpg'.format(orginal_name)
    res = requests.get(url, stream = True)
    if res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image sucessfully Downloaded: ',name)
    else:
        name = name.replace('-', ' ').lower()
        url = "https://img.pokemondb.net/artwork/large/{0}.jpg".format(name)
        file_name = 'pokemon_images/{0}.jpg'.format(orginal_name)
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(file_name, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print('Image sucessfully Downloaded: ', name)
        else:
            words = name.split()
            try:
                name = swap_words(name, str(words[0]), str(words[1]))
                name = name.replace(' ', '-').lower()
            except IndexError:
                if name[len(name)-1] == '♀':
                    name = name[:-1] + '-f'
                elif name[len(name)-1] == '♂':
                    name = name[:-1] + '-m'
            url = "https://img.pokemondb.net/artwork/large/{0}.jpg".format(name)
            file_name = 'pokemon_images/{0}.jpg'.format(orginal_name)
            res = requests.get(url, stream=True)
            if res.status_code == 200:
                with open(file_name, 'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                print('Image sucessfully Downloaded: ', name)
            else:
                try:
                    name = name.replace('-', ' ')
                    words = name.split()
                    name = str(words[1])
                except IndexError:
                    print("Error {0}".format(orginal_name))
                url = "https://img.pokemondb.net/artwork/large/{0}.jpg".format(name)
                file_name = 'pokemon_images/{0}.jpg'.format(orginal_name)
                res = requests.get(url, stream=True)
                if res.status_code == 200:
                    with open(file_name, 'wb') as f:
                        shutil.copyfileobj(res.raw, f)
                    print('Image sucessfully Downloaded: ', name)
                else:
                    error_name.append(name)
print(error_name)