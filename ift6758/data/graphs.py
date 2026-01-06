from ift6758.data import load_cached_season_dataframe, get_player_name
from PIL import Image as PILImage
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

def compute_team_diff(play_df, team_name, x_edges, y_edges, league_average_smoothed, sigma=3):
    team_df = play_df[
        (play_df["teamShot"] == team_name) &
        (play_df["typeEvent"].isin(["shot-on-goal", "goal"]))
    ].copy()

    n_games_team = team_df["gameId"].nunique()

    H_team, _, _ = np.histogram2d(
        team_df["x"],
        team_df["y"],
        bins=[x_edges, y_edges]
    )

    team_rate = H_team / n_games_team
    team_rate_smoothed = gaussian_filter(team_rate, sigma=sigma)

    diff_rate = team_rate_smoothed - league_average_smoothed
    return diff_rate, n_games_team

def plotly_team_dropdown(team_maps, team_games, x_edges, y_edges, rink_data, year):
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2  # 0..100
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2  # -42.5..42.5

    teams = list(team_maps.keys())
    t0 = teams[0]

    # Échelle par équipe (comme matplotlib max_val = max(abs(diff_rate)))
    team_max = {t: float(np.max(np.abs(team_maps[t]))) for t in teams}
    m0 = team_max[t0]

    fig = go.Figure(go.Contour(
        z=team_maps[t0],
        x=y_centers,   # largeur
        y=x_centers,   # distance
        ncontours=15,
        contours=dict(coloring="heatmap", showlines=True),
        line=dict(width=0.5),
        zmin=-m0, zmax=m0,
        colorscale="RdBu",
        reversescale=True,
        opacity=0.75,
        colorbar=dict(title="Excès tirs vs ligue")
    ))

    # Patinoire en fond
    fig.add_layout_image(dict(
        source=rink_data,
        xref="x", yref="y",
        x=-42.5, y=100,
        sizex=85.0, sizey=100.0,
        sizing="stretch",
        layer="below"
    ))

    buttons = []
    for t in teams:
        m = team_max[t]
        buttons.append(dict(
            label=t,
            method="update",
            args=[
                {  # data updates
                    "z": [team_maps[t]],
                    "zmin": [-m],
                    "zmax": [m],
                },
                {  
                "title": {"text": f"Différence de tirs: {t} ({team_games[t]} matchs) — saison {year}-{year+1}"}
            }
            ]
        ))

    fig.update_layout(
        title=f"Différence de tirs: {t0} ({team_games[t0]} matchs) — saison {year}",
        updatemenus=[dict(buttons=buttons, direction="down", x=1.02, y=1.0)],
        xaxis=dict(range=[-42.5, 42.5], title="Largeur (pieds)"),
        yaxis=dict(range=[0, 100], title="Distance (pieds)", scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=180, t=60, b=40),
        height=650
    )
    return fig

def plot_goal_curve(y_true, y_proba, label):
    """Calcule et trace la courbe cumulative pour un modèle donné."""
    # Trier par probabilité décroissante
    order = np.argsort(-y_proba)
    y_sorted = np.asarray(y_true)[order].astype(int)

    # Calculs cumulatifs
    total_goals = y_sorted.sum()
    cum_goals = np.cumsum(y_sorted)

    if total_goals > 0:
        cum_prop_percent = 100.0 * (cum_goals / total_goals)
    else:
        cum_prop_percent = np.zeros_like(cum_goals, dtype=float)

    # Calcul de l'axe X (Percentile)
    pct_shots_selected = (np.arange(1, len(y_sorted) + 1) / len(y_sorted)) * 100.0
    shot_probability_percentile = 100.0 - pct_shots_selected

    # Tracé
    plt.plot(shot_probability_percentile, cum_prop_percent, label=label)