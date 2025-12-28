from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import requests
import json
from pathlib import Path
from IPython.display import clear_output

def get_game(game_year, game_type, game_number):
    """Récupère les infos d'une partie à partir de la chache ou de l'API

    Paramètres:
    game_year (int): Année ou la partie a eu lieu
    game_type (int): Type de la partie (1 = pré-saison, 2 = saison régulière, 3 = playoffs, 4 = all-star)
    game_numer (int): Numéro de la partie

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
    base_path = Path("data") / str_year 
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

def play_context(data):
    """Récupère plusieurs données pour notre outil de déboggage interactif

    Paramètre:
    data: Toutes les données d'une partie

    Retourne:
    home, away, team_by_id, player_name: L'équipe à domicile, l'équipe visiteuse, les noms des deux équipes et le noms des joueurs
    """
    
    home = data["homeTeam"]
    away = data["awayTeam"]
    team_by_id = {home["id"]: home, away["id"]: away}
    player_name = {}
    
    for player in data.get("rosterSpots", []):
        player_id = player.get("playerId")
        first_name = (player.get("firstName") or {}).get("default", "")
        last_name = (player.get("lastName") or {}).get("default", "")
        if player_id is not None:
            player_name[player_id] = (first_name + " " + last_name).strip()

    return home, away, team_by_id, player_name

def get_player_name(player_id):
    """Récupère le nom d'un joueur s'il est connu

    Parmètre:
    player_id (int): ID du joueur

    Retourne:
    player_name (String): Nom du joueur
    
    """
    return player_name.get(player_id, f"#{player_id}" if player_id is not None else "Unknown")
    
def play_description(play, team_by_id, player_name):
    """Retourne les données du jeu selon le type de ce dernier

    Paramètres:
    play (dict): Données complètes du jeu
    team_by_id: Équipes du jeu
    player_name: Joueurs du jeu
    
    """
    type_ofplay = play.get("typeDescKey", "")
    details = play.get("details", {}) or {}

    match type_ofplay:
        case "blocked-shot":
            blocker = get_player_name(details.get('blockingPlayerId'))
            shooter = get_player_name(details.get('shootingPlayerId'))
            return f"{blocker} blocked shot from {shooter}"

        case "shot-on-goal":
            shot_type = details.get("shotType")
            suffix = f" ({shot_type})" if shot_type else ""
            return f"{get_player_name(details.get('shootingPlayerId'))} shot on goal{suffix}"

        case "missed-shot":
            shot_type = details.get("shotType")
            suffix = f" ({shot_type})" if shot_type else ""
            return f"{get_player_name(details.get('shootingPlayerId'))} missed shot{suffix}"

        case "goal":
            scorer = get_player_name(details.get('scoringPlayerId'))
            return f"GOAL — {scorer}"

        case "faceoff":
            winner = get_player_name(details.get('winningPlayerId'))
            return f"Faceoff won by {winner}"

        case "hit":
            hitter = get_player_name(details.get('hittingPlayerId'))
            hittee = get_player_name(details.get('hitteePlayerId'))
            return f"{hitter} hit {hittee}"

        case _:
            return type_ofplay or "event"

def get_goal_strength(play):
    if play.get('typeDescKey') != 'goal':
        return ''
    
    # On récupère le code de situation (ex: "1541")
    situation_code = play.get('situationCode')

    # On parse les chiffres (Attention: l'ordre est Away Goalie, Away Skaters, Home Skaters, Home Goalie)
    # Source: Communauté API NHL (les indices sont 0, 1, 2, 3)
    away_skaters = int(situation_code[1])
    home_skaters = int(situation_code[2])
    
    # On identifie l'équipe qui a marqué
    details = play.get('details', {})
    scoring_team_id = details.get('eventOwnerTeamId')
    
    # On détermine le nombre de patineurs pour l'équipe qui marque vs l'adversaire
    # Note: On utilise les variables globales 'home' et 'away' du contexte
    if scoring_team_id == away.get('id'):
        goals_for = away_skaters
        goals_against = home_skaters
    elif scoring_team_id == home.get('id'):
        goals_for = home_skaters
        goals_against = away_skaters
    else:
        return "Unknown"

    # --- LOGIQUE DE DÉTECTION ---
    
    # 1. Avantage numérique (Power Play)
    # Si j'ai plus de joueurs que l'autre ET que l'autre en a moins de 5 (pour exclure le 6v5 filet désert)
    if (goals_for > goals_against) and (goals_against < 5):
        return "PP"
        
    # 2. Désavantage numérique (Shorthanded)
    # Si j'ai moins de joueurs que l'autre ET que j'en ai moins de 5 (pour exclure le 5v6 filet désert)
    elif (goals_for < goals_against) and (goals_for < 5):
        return "SH"
        
    # 3. Force égale (Even Strength)
    # Inclut le 5v5, 4v4, 3v3 et les buts en filet désert classiques (5v6 ou 6v5 sans pénalité)
    else:
        return "EV"

def create_plays_dataframe(plays):
    types = ['goal', 'shot-on-goal']
    filtered_plays = [p for p in plays if p.get('typeDescKey') in types]


    clean_data = []
    for play in filtered_plays:
        details = play.get('details', {})
        owner_id = details.get('eventOwnerTeamId')
        
        if owner_id == away.get('id'):
            teamShot = away.get('abbrev')
        elif owner_id == home.get('id'):
            teamShot = home.get('abbrev')
        
        shooter_id = details.get('shootingPlayerId') or details.get('scoringPlayerId')
        goalie_id = details.get('goalieInNetId')
        is_empty_net = (play.get('typeDescKey') == 'goal') and (goalie_id is None)
        
        clean_data.append({
            'timeInPeriod': play.get('timeInPeriod'),
            'period': (play.get('periodDescriptor') or {}).get('number', '?'),
            'eventId': play.get('eventId'),
            'teamShot': teamShot,
            'typeEvent': play.get('typeDescKey'),
            'x': details.get('xCoord'), 
            'y': details.get('yCoord'),
            'shooter': get_player_name(shooter_id),
            'goalie': get_player_name(goalie_id),
            'typeShot' : details.get('shotType'),
            'openNet' : is_empty_net,
            'goalStrenght' : get_goal_strength(play)
        })
    
    new_df = pd.DataFrame(clean_data)
    return new_df

def load_full_regular_season_dataframe(year, game_type=2, max_games=1312):
    global data, all_plays, home, away, team_by_id, player_name

    all_game_dfs = []

    for game_number in range(1, max_games + 1):
            data = get_game(year, game_type, game_number)
            if not data or "plays" not in data:
                continue

            all_plays = data["plays"]
            home, away, team_by_id, player_name = play_context(data)

            df_game = create_plays_dataframe(all_plays)

            # Ajoute des colonnes utiles pour filtrer ensuite
            game_id = data.get("id")
            df_game["gameId"] = game_id
            df_game["season"] = int(str(game_id)[:4]) if game_id else None

            all_game_dfs.append(df_game)

    return pd.concat(all_game_dfs, ignore_index=True)

def summarize_shots_and_goals_by_shot_type(season_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Trie le dataframe par type de tir
    Paramètre : Dataframe d'une saison
    
    Sortie: Data frame trié
    """
    shots_and_goals = season_dataframe[
        season_dataframe["typeEvent"].isin(["shot-on-goal", "goal"])
    ].copy()

    shots_and_goals["typeShot"] = shots_and_goals["typeShot"].fillna("Unknown")
    shots_and_goals["isGoal"] = (shots_and_goals["typeEvent"] == "goal").astype(int)

    summary_by_shot_type = (
        shots_and_goals.groupby("typeShot", as_index=False)
        .agg(shots=("typeEvent", "size"), goals=("isGoal", "sum"),)
    )

    summary_by_shot_type["goal_rate"] = summary_by_shot_type["goals"] / summary_by_shot_type["shots"]
    return summary_by_shot_type.sort_values("shots", ascending=False)
