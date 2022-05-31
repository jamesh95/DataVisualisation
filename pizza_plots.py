import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
from mplsoccer import PyPizza, add_image, FontManager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

# This script plots pizza charts detailing the primary and secondary playing style attributes of all teams in Premier League 2021-2022 season

# This removes the 'SettingWithCopyWarning'
pd.set_option('mode.chained_assignment', None)

# Plot primary team styles (focused on passing and possession data)
def plot_primary(dataframe, team_df, average_df, team_name):
    font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/static/Roboto-Regular.ttf?raw=true"))
    font_italic = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/static/Roboto-Italic.ttf?raw=true"))
    font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/static/Roboto-Medium.ttf?raw=true"))
    img = "img/"+team_name+".png"
    team_logo = Image.open(img)

    params = ['Long Passes %', 'Medium Passes %', 'Short Passes %', 'Ground Passes %', 'Low Passes %', 'High Passes %', 'Possession %', 'Directness']
    values = [team_df["long_attem%"].iloc[0], team_df["medium_attem%"].iloc[0], team_df["short_attem%"].iloc[0], team_df["ground%"].iloc[0], team_df["low%"].iloc[0], team_df["high%"].iloc[0], team_df["possession"].iloc[0], team_df["prog_distance%"].iloc[0]]
    values2 = [average_df["long_attem%"].iloc[0], average_df["medium_attem%"].iloc[0], average_df["short_attem%"].iloc[0], average_df["ground%"].iloc[0], average_df["low%"].iloc[0], average_df["high%"].iloc[0], average_df["possession"].iloc[0], average_df["prog_distance%"].iloc[0]]

    min_range = [dataframe["long_attem%"].min()-5, dataframe["medium_attem%"].min()-5, dataframe["short_attem%"].min()-5, dataframe["ground%"].min()-5, dataframe["low%"].min()-5, dataframe["high%"].min()-5, dataframe["possession"].min()-5, dataframe["prog_distance%"].min()-5]
    max_range = [dataframe["long_attem%"].max()+2, dataframe["medium_attem%"].max()+2, dataframe["short_attem%"].max()+2, dataframe["ground%"].max()+2, dataframe["low%"].max()+2, dataframe["high%"].max()+2, dataframe["possession"].max()+2, dataframe["prog_distance%"].max()+2]

    slice_colors = ["#227c9d"] * 3 + ["#17c3b2"] * 3 + ["#ffcb77"] + ["#fef9ef"]

    pizza = PyPizza(
        params=params,                  # list of parameters
        background_color="#111111",     # background color
        min_range = min_range,
        max_range = max_range,
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=0,               # linewidth of last circle
        inner_circle_size=10,            # increase the circle size
        other_circle_lw=0,             # linewidth for other circles
        #other_circle_ls="-."            # linestyle for other circles
        )

    fig, ax = pizza.make_pizza(
    values,              # list of values
    compare_values=values2,    # passing comparison values
    figsize=(8, 8),      # adjust figsize according to your need
    slice_colors=slice_colors,       # color for individual slices
    color_blank_space="same",   # use same color to fill blank space
    #blank_alpha=0.4,                 # alpha for blank-space colors
    param_location=105,  # where the parameters will be added
    kwargs_slices=dict(
        edgecolor="#000000", zorder=2, linewidth=1
    ),                   # values to be used when plotting slices
    kwargs_compare=dict(
        facecolor="none", edgecolor="#000000", zorder=3, linewidth=1,
    ),                          # values to be used when plotting comparison slices
    kwargs_params=dict(
        color="white", fontsize=12,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="white", fontsize=12,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="black",
            boxstyle="round,pad=0.2", lw=1
            )
    ),                   # values to be used when adding parameter-values
    kwargs_compare_values=dict(
        color="none", fontsize=0,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="none", facecolor="none",
            boxstyle="round,pad=0.2", lw=0
        )
    )                   # values to be used when adding comparison-values
    )
    # add title
    fig.text(
        0.515, 0.97, team_name+" FC", size=18,
        ha="center", fontproperties=font_bold.prop, color="white"
    )
    # add subtitle
    fig.text(
        0.515, 0.942,
        "Primary Attributes vs League Median | Season 2021-22",
        size=15,
        ha="center", fontproperties=font_bold.prop, color="white"
    )
    # add credits
    CREDIT_1 = "data: statsbomb viz fbref"
    CREDIT_2 = "YOUR_NAME"

    fig.text(
        0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
        fontproperties=font_italic.prop, color="white",
        ha="right"
    )
    # add image
    ax_image = add_image(
        team_logo, fig, left=0.4478, bottom=0.4315, width=0.13, height=0.127
    )   # these values might differ when you are plotting

    plt.savefig('PlayStyles/'+team_name+'_primary.png', pad_inches = 0.2, dpi=200, facecolor='#111111')
    plt.show()

# Plot secondary team styles
def plot_secondary(dataframe, team_df, average_df, team_name):
    font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/static/Roboto-Regular.ttf?raw=true"))
    font_italic = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/static/Roboto-Italic.ttf?raw=true"))
    font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/static/Roboto-Medium.ttf?raw=true"))
    img = "img/"+team_name+".png"
    team_logo = Image.open(img)

    params = ['Attacking Pressures p90', 'Cross %', 'Physicality Rating', 'Set Piece Chances', 'NP Goals-xG', 'GK Long Pass %']
    values = [team_df["press_att_p90"].iloc[0], team_df["cross%"].iloc[0], team_df["physicality"].iloc[0], team_df["dead_balls%"].iloc[0], team_df["np_goals-xG"].iloc[0], team_df["long%"].iloc[0]]
    values2 = [average_df["press_att_p90"].iloc[0], average_df["cross%"].iloc[0], average_df["physicality"].iloc[0], average_df["dead_balls%"].iloc[0], average_df["np_goals-xG"].iloc[0], average_df["long%"].iloc[0]]

    min_range = [dataframe["press_att_p90"].min()-5, dataframe["cross%"].min()-1, dataframe["physicality"].min()-5, dataframe["dead_balls%"].min()-5, dataframe["np_goals-xG"].min()-5, dataframe["long%"].min()-5]
    max_range = [dataframe["press_att_p90"].max()+2, dataframe["cross%"].max()+0.5, dataframe["physicality"].max()+2, dataframe["dead_balls%"].max()+2, dataframe["np_goals-xG"].max()+2, dataframe["long%"].max()+2]

    slice_colors = ["#005f73"] + ["#0a9396"] + ["#94d2bd"] + ["#ee9b00"] + ["#ca6702"] + ["#ae2012"]

    pizza = PyPizza(
        params=params,                  # list of parameters
        background_color="#111111",     # background color
        min_range = min_range,
        max_range = max_range,
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=0,               # linewidth of last circle
        inner_circle_size=10,            # increase the circle size
        other_circle_lw=0,             # linewidth for other circles
        #other_circle_ls="-."            # linestyle for other circles
        )

    fig, ax = pizza.make_pizza(
    values,              # list of values
    compare_values=values2,    # passing comparison values
    figsize=(8, 8),      # adjust figsize according to your need
    slice_colors=slice_colors,       # color for individual slices
    color_blank_space="same",   # use same color to fill blank space
    #blank_alpha=0.4,                 # alpha for blank-space colors
    param_location=105,  # where the parameters will be added
    kwargs_slices=dict(
        edgecolor="#000000", zorder=2, linewidth=1
    ),                   # values to be used when plotting slices
    kwargs_compare=dict(
        facecolor="none", edgecolor="#000000", zorder=3, linewidth=1,
    ),                          # values to be used when plotting comparison slices
    kwargs_params=dict(
        color="white", fontsize=12,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="white", fontsize=12,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="black",
            boxstyle="round,pad=0.2", lw=1
            )
    ),                   # values to be used when adding parameter-values
    kwargs_compare_values=dict(
        color="none", fontsize=0,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="none", facecolor="none",
            boxstyle="round,pad=0.2", lw=0
        )
    )                   # values to be used when adding comparison-values
    )
    # add title
    fig.text(
        0.515, 0.97, team_name+" FC", size=18,
        ha="center", fontproperties=font_bold.prop, color="white"
    )
    # add subtitle
    fig.text(
        0.515, 0.942,
        "Secondary Attributes vs League Median | Season 2021-22",
        size=15,
        ha="center", fontproperties=font_bold.prop, color="white"
    )
    # add credits
    CREDIT_1 = "data: statsbomb viz fbref"
    CREDIT_2 = "YOUR_NAME"

    fig.text(
        0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
        fontproperties=font_italic.prop, color="white",
        ha="right"
    )
    # add image
    ax_image = add_image(
        team_logo, fig, left=0.4478, bottom=0.4315, width=0.13, height=0.127
    )   # these values might differ when you are plotting

    plt.savefig('PlayStyles/'+team_name+'_secondary.png', pad_inches = 0.2, dpi=200, facecolor='#111111')
    plt.show()

def pass_distance():
    df = pd.read_csv("passing_data.csv")
    passing_types_df = df[['team', 'completed', 'attempted', 'completion%', 'total_distance', 'prog_distance',
                    'short_completed', 'short_attem','short_comp%', 'med_completed', 'medium_attem', 'medium_comp%',
                    'long_completed', 'long_attem', 'long_comp%']]
    passes_short_perc = []
    passes_medium_perc = []
    passes_long_perc = []
    for index, row in passing_types_df.iterrows():
        total_passes_att = row.short_attem+row.medium_attem+row.long_attem
        num = (row.short_attem/total_passes_att)*100
        passes_short_perc.append(round(num, 1))
        num = (row.medium_attem/total_passes_att)*100
        passes_medium_perc.append(round(num, 1))
        num = (row.long_attem/total_passes_att)*100
        passes_long_perc.append(round(num, 1))
    passing_types_df["short_attem%"] = passes_short_perc
    passing_types_df["medium_attem%"] = passes_medium_perc
    passing_types_df["long_attem%"] = passes_long_perc
    average_league_passes_short_perc = passing_types_df["short_attem%"].mean()
    average_league_passes_medium_perc = passing_types_df["medium_attem%"].mean()
    average_league_passes_long_perc = passing_types_df["long_attem%"].mean()
    short_passes_ranked = passing_types_df.sort_values(by='short_attem%', ascending=False)
    medium_passes_ranked = passing_types_df.sort_values(by='medium_attem%', ascending=False)
    long_passes_ranked = passing_types_df.sort_values(by='long_attem%', ascending=False)
    passing_types_df = passing_types_df[['team', 'long_attem%', 'medium_attem%', 'short_attem%']]
    return passing_types_df

def pass_styles():
    df = pd.read_csv("pass_types_data.csv")
    passing_styles_df = df[['team', 'attempted', 'ground', 'low', 'high']]
    passes_ground_perc = []
    passes_low_perc = []
    passes_high_perc = []
    for index, row in passing_styles_df.iterrows():
        num = (row.ground/row.attempted)*100
        passes_ground_perc.append(round(num, 1))
        num = (row.low/row.attempted)*100
        passes_low_perc.append(round(num, 1))
        num = (row.high/row.attempted)*100
        passes_high_perc.append(round(num, 1))
    passing_styles_df["ground%"] = passes_ground_perc
    passing_styles_df["low%"] = passes_low_perc
    passing_styles_df["high%"] = passes_high_perc
    average_league_passes_ground_perc = passing_styles_df["ground%"].mean()
    average_league_passes_low_perc = passing_styles_df["low%"].mean()
    average_league_passes_high_perc = passing_styles_df["high%"].mean()
    ground_passes_ranked = passing_styles_df.sort_values(by='ground%', ascending=False)
    low_passes_ranked = passing_styles_df.sort_values(by='low%', ascending=False)
    high_passes_ranked = passing_styles_df.sort_values(by='high%', ascending=False)
    passing_styles_df = passing_styles_df[['team', 'ground%', 'low%', 'high%']]
    return passing_styles_df

def possession_types():
    df = pd.read_csv("possession_data.csv")
    possession_types_df = df[['team', 'possession']]
    possession_types_df = possession_types_df.sort_values(by='possession', ascending=False)
    possession_types = []
    possession_score = []
    for i in range(len(possession_types_df)):
        if i < 6:
            possession_types.append("high")
            possession_score.append(4)
        elif i >= 6 and i < 11:
            possession_types.append("high-medium")
            possession_score.append(3)
        elif i >= 11 and i < 16:
            possession_types.append("medium-low")
            possession_score.append(2)
        elif i >= 16:
            possession_types.append("low")
            possession_score.append(1)
    possession_types_df["possession_type"] = possession_types
    possession_types_df["possession_score"] = possession_score
    average_league_possession = possession_types_df["possession"].mean()
    possession_types_df["possession_diff"] = possession_types_df["possession"]-average_league_possession
    possession_types_df = possession_types_df[['team', 'possession_type', 'possession', 'possession_diff']]
    return possession_types_df

def possession_styles():
    df = pd.read_csv("passing_data.csv")
    possession_styles_df = df[['team', 'completed', 'attempted', 'total_distance', 'prog_distance', 'prog_passes']]
    prog_distance_perc = []
    prog_passes_perc = []
    for index, row in possession_styles_df.iterrows():
        num = (row.prog_distance/row.total_distance)*100
        prog_distance_perc.append(round(num, 1))
        num = (row.prog_passes/row.completed)*100
        prog_passes_perc.append(round(num, 1))
    possession_styles_df["prog_distance%"] = prog_distance_perc
    possession_styles_df["prog_passes%"] = prog_passes_perc
    average_league_prog_distance_perc = possession_styles_df["prog_distance%"].mean()
    average_league_prog_passes_perc = possession_styles_df["prog_passes%"].mean()
    prog_distance_ranked = possession_styles_df.sort_values(by='prog_distance%', ascending=False)
    prog_passes_ranked = possession_styles_df.sort_values(by='prog_passes%', ascending=False)
    possession_styles_df = possession_styles_df[['team', 'prog_distance%', 'prog_passes%']]
    return possession_styles_df

def high_press():
    df = pd.read_csv("defensive_actions_data.csv")
    high_press_df = df[['team','90s', 'pressures', 'press_succ', 'press_succ%', 'press_def', 'press_mid', 'press_att']]
    press_att_perc = []
    for index, row in high_press_df.iterrows():
        num = (row.press_att/row.pressures)*100
        press_att_perc.append(round(num, 1))
    high_press_df["press_att%"] = press_att_perc
    average_press_att = high_press_df["press_att%"].mean()
    high_press_ranked = high_press_df.sort_values(by='press_att', ascending=False)
    high_press_df["press_att_p90"] = high_press_df["press_att"]/high_press_df["90s"]
    high_press_df = round(high_press_df[['team', 'press_att', 'press_att_p90']],1)
    return high_press_df

def crossing():
    df = pd.read_csv("pass_types_data.csv")
    crossing_df = df[['team', 'attempted', 'cross']]
    crossing_perc = []
    for index, row in crossing_df.iterrows():
        num = (row.cross/row.attempted)*100
        crossing_perc.append(round(num, 1))
    crossing_df["cross%"] = crossing_perc
    average_crosses = crossing_df["cross%"].mean()
    crosses_ranked = crossing_df.sort_values(by='cross%', ascending=False)
    crossing_df = crossing_df[['team', 'cross%']]
    return crossing_df

def physicality():
    df = pd.read_csv("misc_data.csv")
    misc_df = df[['team', 'yellows', 'reds', 'fouls', 'aerials_won', 'aerials_lost', 'aerials_won%']]
    aerials_ranked = misc_df.sort_values(by='aerials_won%', ascending=False)
    fouls_ranked = misc_df.sort_values(by='fouls', ascending=False)
    df = pd.read_csv("defensive_actions_data.csv")
    tackles_df = df[['team', 'tackles', 'tackles_won']]
    tackles_won_perc = []
    for index, row in tackles_df.iterrows():
        num = (row.tackles_won/row.tackles)*100
        tackles_won_perc.append(round(num, 1))
    tackles_df["tackles_won%"] = tackles_won_perc
    average_tackles_won = tackles_df["tackles_won%"].mean()
    tackles_ranked = tackles_df.sort_values(by='tackles_won%', ascending=False)
    misc_df["tackles"] = tackles_df.tackles
    misc_df["tackles_won"] = tackles_df.tackles_won
    misc_df["tackles_won%"] = tackles_df["tackles_won%"]
    misc_df = misc_df[['team', 'fouls', 'aerials_won%', 'tackles_won%']]
    fouls_norm = round((1 + misc_df.fouls/misc_df.fouls.max()*9)*10,1)
    aerials_won_norm = round((1 + misc_df['aerials_won%']/misc_df['aerials_won%'].max()*9)*10,1)
    tackles_won_norm = round((1 + misc_df['tackles_won%']/misc_df['tackles_won%'].max()*9)*10,1)
    physicality_score = round((fouls_norm+aerials_won_norm+tackles_won_norm)/3,1)
    misc_df["physicality"] = physicality_score
    return misc_df

def set_pieces():
    df = pd.read_csv("goal_shot_creation_data.csv")
    dead_balls_df = df[['team', 'shot_cre_acts', 'dead_sca']]
    dead_balls_perc = []
    for index, row in dead_balls_df.iterrows():
        num = (row.dead_sca/row.shot_cre_acts)*100
        dead_balls_perc.append(round(num, 1))
    dead_balls_df["dead_balls%"] = dead_balls_perc
    average_dead_balls = dead_balls_df["dead_balls%"].mean()
    dead_balls_ranked = dead_balls_df.sort_values(by='dead_balls%', ascending=False)
    dead_balls_df = dead_balls_df[['team', 'dead_balls%']]
    return dead_balls_df

def shooting():
    df = pd.read_csv("shooting_data.csv")
    shooting_df = df[['team','shots_p90', 'avg_dist', 'np_goals-xG']]
    distance_ranked = shooting_df.sort_values(by='avg_dist', ascending=False)
    shots_ranked = shooting_df.sort_values(by='shots_p90', ascending=False)
    shooting_ranked = shooting_df.sort_values(by='np_goals-xG', ascending=False)
    return shooting_df

def play_out():
    df = pd.read_csv("goalkeepers_adv.csv")
    gk_df = df[['team', 'long_pass%', 'gk_long%', 'gk_avg_len']]
    longgk_ranked = gk_df.sort_values(by='gk_long%', ascending=False)
    longpass_ranked = gk_df.sort_values(by='long_pass%', ascending=False)
    gk_df["long%"] = round((gk_df["gk_long%"] + gk_df["long_pass%"])/2, 1)
    return gk_df

# Gather primary and secondary features from the csv files in the project folder
def playstyles_data():
    pass_types_df = pass_distance()
    pass_styles_df = pass_styles()
    possession_types_df = possession_types()
    possession_styles_df = possession_styles()
    high_press_df = high_press()
    crossing_df = crossing()
    physicality_df = physicality()
    set_pieces_df = set_pieces()
    shooting_df = shooting()
    play_out_df = play_out()
    play_styles_df = pd.concat([pass_types_df, pass_styles_df, possession_types_df, possession_styles_df, high_press_df, crossing_df, physicality_df, set_pieces_df, shooting_df, play_out_df], axis=1)
    play_styles_df = play_styles_df.loc[:,~play_styles_df.columns.duplicated()]
    return play_styles_df

# Plot styles for a given team
def plot_style(team_name, name):
    play_styles_df = playstyles_data()
    team_styles_df = play_styles_df.loc[play_styles_df.team == name]
    average_df = pd.DataFrame()
    average_df["long_attem%"] = [play_styles_df["long_attem%"].median()]
    average_df["medium_attem%"] = play_styles_df["medium_attem%"].median()
    average_df["short_attem%"] = play_styles_df["short_attem%"].median()
    average_df["ground%"] = play_styles_df["ground%"].median()
    average_df["low%"] = play_styles_df["low%"].median()
    average_df["high%"] = play_styles_df["high%"].median()
    average_df["possession"] = play_styles_df["possession"].median()
    average_df["prog_distance%"] = play_styles_df["prog_distance%"].median()
    average_df["press_att_p90"] = play_styles_df["press_att_p90"].median()
    average_df["cross%"] = play_styles_df["cross%"].median()
    average_df["physicality"] = play_styles_df["physicality"].median()
    average_df["dead_balls%"] = play_styles_df["dead_balls%"].median()
    average_df["np_goals-xG"] = play_styles_df["np_goals-xG"].median()
    average_df["long%"] = play_styles_df["long%"].median()
    plot_primary(play_styles_df, team_styles_df, average_df, team_name)
    plot_secondary(play_styles_df, team_styles_df, average_df, team_name)

# Plot styles for all teams
def plot_styles_for_teams():
    names = ["Arsenal", "Aston Villa", "Brentford", "Brighton", "Burnley", "Chelsea", "Crystal Palace", "Everton", "Leeds United", "Leicester City", "Liverpool", "Manchester City", "Manchester Utd", "Newcastle Utd", "Norwich City", "Southampton", "Tottenham", "Watford", "West Ham", "Wolves"]
    file_names = ["Arsenal", "AstonVilla", "Brentford", "Brighton", "Burnley", "Chelsea", "CrystalPalace", "Everton", "Leeds", "Leicester", "Liverpool", "ManCity", "ManUtd", "Newcastle", "Norwich", "Southampton", "Spurs", "Watford", "WestHam", "Wolves"]

    for i in range(len(names)):
        plot_style(file_names[i], names[i])
