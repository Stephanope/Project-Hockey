from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import requests
import json
from typing import Iterator
from pathlib import Path
from IPython.display import clear_output
import os
from typing import Iterable, Union

def get_data_dir() -> Path:
    # 1) variable d'env (meilleur)
    env = os.environ.get("NHL_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # 2) fallback : racine du repo -> /data
    # data_exploration.py est dans .../Project-Hockey/ift6758/data/data_exploration.py
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "data").resolve()

def get_game(game_year, game_type, game_number):
    """Récupère les infos d'une partie à partir de la chache ou de l'API

    Paramètres:
    game_year (int): Année ou la partie a eu lieu
    game_type(int): Type de la partie (1 = pré-saison, 2 = saison régulière, 3 = playoffs, 4 = all-star)
    game_numer (in t): Numéro de la partie

    Retourne:
    data (dict): Données de la partie spécifiée
    """

    # Conversion des int en string pour l'API
    str_year = str(game_year)
    str_type = str(game_type).zfill(2) 
    str_num = str(game_number).zfill(4) 

    # Création de la requête GET
    game_ID = str_year + str_type + str_num 
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_ID}/play-by-play"  

    # Création du path du fichier qui va stocker les données
    file_name = game_ID + ".json" 
    base_path = get_data_dir() / str_year 
    base_path.mkdir(parents=True, exist_ok=True) 
    complete_path = base_path / file_name 

    # Si les données sont présentes localement on les récupère directement
    if complete_path.is_file():
        with open(complete_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Sinon on fait une requête à l'API de la NHL
    else:   
        response = requests.get(url) 
        if response.status_code == 200: 
            data = response.json()     
            with open(complete_path, 'w', encoding='utf-8') as f: 
               json.dump(data, f, indent=4) 
            return data
        else: 
            print(f"Erreur {response.status_code} lors du téléchargement.") 
            return None  


def fetch_full_year_regular(year):
    """Récupère les données des parties d'une saison régulière complète

    Paramètre:
    year (int): Année de la saison régulière
    """
    for i in range(1, 1351): 
        data = get_game(year, 2, i) 
        if data is None:
            print(f"Arrêt à la partie {i}, fin de la saison détectée.")
            break

def fetch_all_seasons_regular(start_year: int = 2016, end_year: int = 2023):
    """
    Récupère les données des parties de toutes les saisons de start_year jusqu'à end_year
    Paramètre:
    start_year (int): première saison qu'on fetch
    end_year (int): dernière saison qu'on fetch
    """
    for year in range(start_year, end_year + 1):
        fetch_full_year_regular(year)
    
def fetch_full_year_playoffs(year: int):
    # séries “standards” : R1=8, R2=4, R3=2, Final=1
    series_per_round = {1: 8, 2: 4, 3: 2, 4: 1}

    for rnd, n_series in series_per_round.items():
        for series in range(1, n_series + 1):
            for game in range(1, 8):  # best-of-7
                game_number = int(f"{rnd:02d}{series}{game}")  # 0111, 0112, ..., 0477
                data = get_game(year, 3, game_number)

                if data is None:
                    break

def fetch_all_seasons_playoffs(start_year: int = 2016, end_year: int = 2023):
    """
    Récupère les données des parties des playoffs de toutes les saisons de start_year jusqu'à end_year
    Paramètre:
    start_year (int): première saison qu'on fetch
    end_year (int): dernière saison qu'on fetch
    """
    for year in range(start_year, end_year + 1):
        fetch_full_year_playoffs(year)