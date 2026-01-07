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
from ift6758.data.nhl_api import get_data_dir
import math

DATA_DIR = get_data_dir()

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

def get_player_name(player_id, player_name: dict):
    """Récupère le nom d'un joueur s'il est connu

    Parmètre:
    player_id (int): ID du joueur

    Retourne:
    player_name (String): Nom du joueur
    
    """
    return player_name.get(player_id, f"#{player_id}" if player_id is not None else "Unknown")
    
def play_description(play, team_by_id, player_name):
    type_ofplay = play.get("typeDescKey", "")
    details = play.get("details", {}) or {}

    match type_ofplay:
        case "blocked-shot":
            blocker = get_player_name(details.get('blockingPlayerId'), player_name)
            shooter = get_player_name(details.get('shootingPlayerId'), player_name)
            return f"{blocker} blocked shot from {shooter}"

        case "shot-on-goal":
            shot_type = details.get("shotType")
            suffix = f" ({shot_type})" if shot_type else ""
            return f"{get_player_name(details.get('shootingPlayerId'), player_name)} shot on goal{suffix}"

        case "missed-shot":
            shot_type = details.get("shotType")
            suffix = f" ({shot_type})" if shot_type else ""
            return f"{get_player_name(details.get('shootingPlayerId'), player_name)} missed shot{suffix}"

        case "goal":
            scorer = get_player_name(details.get('scoringPlayerId'), player_name)
            return f"GOAL — {scorer}"

        case "faceoff":
            winner = get_player_name(details.get('winningPlayerId'), player_name)
            return f"Faceoff won by {winner}"

        case "hit":
            hitter = get_player_name(details.get('hittingPlayerId'), player_name)
            hittee = get_player_name(details.get('hitteePlayerId'), player_name)
            return f"{hitter} hit {hittee}"

        case _:
            return type_ofplay or "event"

def get_goal_strength(play):
    
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


def time_between_events(current_play, last_play):
    """
    Retourne le temps (en secondes) entre deux événements
    supposés être dans la même période.
    """

    def to_seconds(play):
        minutes, seconds = map(int, play["timeInPeriod"].split(":"))
        return minutes * 60 + seconds

    return to_seconds(current_play) - to_seconds(last_play)

def distance_between_events(current_play, last_play):
    """
    Retourne la distance (en pieds) entre deux événements NHL.
    Retourne None si les coordonnées sont manquantes.
    """

    try:
        x1 = last_play["details"]["xCoord"]
        y1 = last_play["details"]["yCoord"]
        x2 = current_play["details"]["xCoord"]
        y2 = current_play["details"]["yCoord"]
    except KeyError:
        return None

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    

def create_plays_dataframe(plays, player_name):
    types = ['goal', 'shot-on-goal']
    filtered_plays = [p for p in plays if p.get('typeDescKey') in types]

    clean_data = []
    for i, play in enumerate(plays):
        if play in filtered_plays:
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
                'shooter': get_player_name(shooter_id, player_name),
                'goalie': get_player_name(goalie_id, player_name),
                'typeShot' : details.get('shotType'),
                'openNet' : is_empty_net,
                'goalStrenght' : get_goal_strength(play),
                'lastEvent' : plays[i-1].get('typeDescKey'),
                'lastEventX' : plays[i-1].get('details', {}).get('xCoord'),
                'lastEventY' : plays[i-1].get('details', {}).get('yCoord'),
                'timeSinceLastEvent' : time_between_events(play, plays[i-1]),
                'distanceSinceLastEvent' : distance_between_events(play, plays[i-1]),
            })
    
    new_df = pd.DataFrame(clean_data)
    return new_df

    
def iter_cached_games(year: int, game_type: int = 2):
    base = DATA_DIR / str(year)
    pattern = f"{year}{str(game_type).zfill(2)}*.json"
    for p in sorted(base.glob(pattern)):
        with open(p, "r", encoding="utf-8") as f:
            yield json.load(f)


def load_cached_plays_raw_dataframe(year: int, game_type: int = 2, max_games: int | None = None) -> pd.DataFrame:
    rows = []
    for k, game in enumerate(iter_cached_games(year, game_type), start=1):
        if max_games is not None and k > max_games:
            break
        game_id = game.get("id")
        for play in (game.get("plays") or []):
            d = play.get("details") or {}
            rows.append({
                "gameId": game_id,
                "eventId": play.get("eventId"),
                "typeDescKey": play.get("typeDescKey"),
                "period": (play.get("periodDescriptor") or {}).get("number"),
                "timeInPeriod": play.get("timeInPeriod"),
                "teamId": d.get("eventOwnerTeamId"),
                "x": d.get("xCoord"),
                "y": d.get("yCoord"),
                "shotType": d.get("shotType"),
            })
    return pd.DataFrame(rows)

def load_cached_season_dataframe(year: int, game_types: Union[int, Iterable[int]] = (2, 3)) -> pd.DataFrame:
    """
    Charge une saison depuis les JSON cachés et retourne un DF concaténé.
    game_types: 2 (saison), 3 (playoffs), ou (2,3) pour les deux.
    """
    # Permet de passer soit un int, soit une liste/tuple
    if isinstance(game_types, int):
        game_types = (game_types,)

    dfs = []

    for gt in game_types:
        for game in iter_cached_games(year, game_type=gt):
            if not game or "plays" not in game:
                continue

            global home, away, team_by_id, player_name
            home, away, team_by_id, player_name = play_context(game)

            df_game = create_plays_dataframe(game["plays"], player_name)
            df_game["gameId"] = game.get("id")
            df_game["season"] = year
            df_game["gameType"] = gt  # IMPORTANT: on met le bon type ici

            dfs.append(df_game)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_cached_seasons_dataframe(start_year: int = 2016, end_year: int = 2023, game_types: Union[int, Iterable[int]] = (2, 3)) -> pd.DataFrame:
    dfs = []
    for y in range(start_year, end_year + 1):
        df_y = load_cached_season_dataframe(y, game_types=game_types)
        if not df_y.empty:
            dfs.append(df_y)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Pour le notebook de visualisation

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

def add_attack_sign(df,
                    x_col="x", team_col="teamShot",
                    game_col="gameId", period_col="period",
                    event_col="typeEvent"):
    ndf = df.copy()

    shots = ndf[
        ndf[event_col].isin({"shot-on-goal", "goal", "missed-shot"})
        & ndf[x_col].notna()
    ].copy()

    attack = (
        shots.groupby([game_col, period_col, team_col])[x_col]
        .median()
        .gt(0)
        .astype(int)
        .replace({0: -1, 1: 1})
        .rename("attack_sign")
        .reset_index()
    )

    ndf = ndf.merge(attack, on=[game_col, period_col, team_col], how="left")

    ndf["attack_sign"] = (
        ndf.groupby([game_col, team_col])["attack_sign"]
        .transform(lambda s: s.ffill().bfill())
    )

    return ndf

def bin_midpoint(interval):
    return (interval.left + interval.right) / 2

def calculate_angle_between_shots(df, goal_x=89.0, goal_y=0.0):
    """
    Même calcul, mais effectué ligne par ligne avec une boucle.
    """
    angles = []

    # On itère sur chaque ligne du DataFrame (index, contenu de la ligne)
    for index, row in df.iterrows():

        if row["isRebound"] == 0:
            angles.append(0.0)
        else:
            # 1. Vecteur dernier event -> but
            v1x = goal_x - row["lastEventX"]
            v1y = row["lastEventY"] - goal_y  # Note: Je garde ta logique de signe originale
    
            # 2. Vecteur tir courant -> but
            v2x = goal_x - row["x_adj"]
            v2y = row["y_adj"] - goal_y
    
            # 3. Calculs intermédiaires (Produit scalaire et Normes)
            dot = v1x * v2x + v1y * v2y
            norm1 = np.sqrt(v1x**2 + v1y**2)
            norm2 = np.sqrt(v2x**2 + v2y**2)
    
            # 4. Calcul de l'angle avec sécurité
            # Si une des normes est 0 (ex: tir depuis le centre du filet), on évite la division par 0
            if norm1 == 0 or norm2 == 0:
                angles.append(0.0)
            else:
                cos_val = dot / (norm1 * norm2)
                
                # Clip pour éviter les erreurs numériques (ex: 1.000000002 qui fait planter arccos)
                if cos_val > 1.0:
                    cos_val = 1.0
                elif cos_val < -1.0:
                    cos_val = -1.0
                    
                angle_rad = np.arccos(cos_val)
                angle_deg = np.degrees(angle_rad)
                angles.append(angle_deg)

    # Une fois la boucle finie, on assigne la liste à la nouvelle colonne
    return angles

def new_variables(df, goal_x=89.0, goal_y=0.0,
                  x_col="x", y_col="y",
                  event_col="typeEvent", empty_col="openNet"):
    ndf = add_attack_sign(df, x_col=x_col, team_col="teamShot",
                          game_col="gameId", period_col="period", event_col=event_col)

    ndf["x_adj"] = ndf[x_col] * ndf["attack_sign"]
    ndf["y_adj"] = ndf[y_col] * ndf["attack_sign"]

    dx = goal_x - ndf["x_adj"]
    dy = ndf["y_adj"] - goal_y

    ndf["shotDistance"] = np.sqrt(dx**2 + dy**2)
    ndf["shotAngle"] = np.degrees(np.arctan2(dy.abs(), dx))
    ndf["isGoal"] = ndf[event_col].astype(str).str.lower().eq("goal")
    ndf["isEmpty"] = ndf[empty_col].fillna(False).astype(int).eq(1)
    ndf["isRebound"] = ((ndf["lastEvent"] == "shot-on-goal") & (ndf["timeSinceLastEvent"] <= 5)).astype(int)    
    ndf["angleDifference"] = calculate_angle_between_shots(ndf)
    ndf["speed"] = df["distanceSinceLastEvent"] / df["timeSinceLastEvent"]

    return ndf

def goal_rate_by_percentile(y_true, proba_goal, step=5):
    validation_df = pd.DataFrame({
        "is_goal": np.asarray(y_true).astype(int),
        "proba_goal": np.asarray(proba_goal),
    })

    validation_df["percentile"] = validation_df["proba_goal"].rank(pct=True) * 100.0

    percentile_bins = np.arange(0, 100 + step, step)
    validation_df["percentile_bin"] = pd.cut(
        validation_df["percentile"],
        bins=percentile_bins,
        include_lowest=True,
        right=False
    )

    goal_rate_table = (
        validation_df
        .groupby("percentile_bin", observed=True)["is_goal"]
        .mean()
        .reset_index(name="goal_rate")
    )

    bin_midpoints = np.array([
        interval.left + step / 2
        for interval in goal_rate_table["percentile_bin"]
    ])

    return bin_midpoints, goal_rate_table["goal_rate"].to_numpy()