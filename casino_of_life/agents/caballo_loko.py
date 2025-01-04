import os
from character_package.character import Character

 # Path to the JSON file, relative to this file
json_path = os.path.join(os.path.dirname(__file__), 'character_package', 'character.json')

try:
    caballo_loko = Character(json_path)
    print(caballo_loko)
    print(f"Bio: {caballo_loko.bio}")
    print(f"Topics: {caballo_loko.topics[0:5]}")
    print(f"Style (chat): {caballo_loko.get_style('chat')}")
    print(f"Style (post): {caballo_loko.get_style('post')}")

except FileNotFoundError as e:
    print(f"File Error: {e}")
except ValueError as e:
    print(f"Value Error: {e}")