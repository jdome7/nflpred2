import nflreadpy as nfl
import pandas as pd
import numpy as np

def get_updated_data():
    windows = [3, 5, 10, 15] 

    years = list(range(2016, 2026))

    numeric_exclude = ['season', 'week']
    exclude = numeric_exclude + ['team', 'opponent_team']

    #Basic player stats
    player_stats = nfl.load_player_stats(years).to_pandas()
    player_keep = ['player_id', 'player_display_name', 'season', 'week', 'position', 'team', 'opponent_team', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered', 'sack_yards_lost', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards', 'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'passing_cpoe', 'pacr', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa', 'fantasy_points']
    player_stats = player_stats[player_keep]
    player_stats = player_stats[player_stats['position'] == 'QB']
    player_stats = player_stats.sort_values(['player_id', 'season', 'week'])
    player_stats['fumbles_lost'] = player_stats['sack_fumbles_lost'] + player_stats['rushing_fumbles_lost']
    player_stats.drop(columns=['sack_fumbles_lost', 'rushing_fumbles_lost'], inplace=True)

    #Next gen player stats
    next_gen = nfl.load_nextgen_stats(years).to_pandas()
    next_gen_keep = ['season', 'week', 'avg_time_to_throw', 'avg_completed_air_yards', 'avg_intended_air_yards', 'avg_air_yards_differential', 'aggressiveness', 'max_completed_air_distance', 'avg_air_yards_to_sticks', 'passer_rating', 'completion_percentage', 'expected_completion_percentage', 'avg_air_distance', 'max_air_distance', 'player_gsis_id']
    next_gen = next_gen[next_gen_keep]
    next_gen = next_gen.sort_values(['player_gsis_id', 'season', 'week'])

    #Schedule dataframe
    schedule = nfl.load_schedules(years).to_pandas()
    schedule_keep = ['game_id', 'season', 'week', 'away_team', 'home_team', 'total_line', 'div_game', 'temp', 'wind', 'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name']
    schedule = schedule[schedule_keep]

    #team stats (to be used for player)
    team_stats = nfl.load_team_stats(years).to_pandas()
    team_stats_keep = ['season', 'week', 'team', 'opponent_team', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa', 'receiving_epa', 'def_tackles_with_assist', 'def_tackles_for_loss', 'def_fumbles_forced', 'def_sacks', 'def_qb_hits', 'def_interceptions', 'def_pass_defended', 'def_tds', 'misc_yards', 'penalty_yards', 'fg_made', 'pat_att']
    team_stats = team_stats[team_stats_keep]
    team_stats = team_stats.sort_values(['team', 'season', 'week'])

    team_stats.rename(columns={col: f'team_{col}' for col in team_stats.columns if col not in exclude}, inplace=True)

    home_players = schedule[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_qb_id', 'home_qb_name', 'total_line', 'temp', 'wind']]
    home_players.columns = ['game_id', 'season', 'week', 'team', 'opponent', 'player_id', 'player_name', 'line', 'temp', 'wind']


    away_players = schedule[['game_id', 'season', 'week', 'away_team', 'home_team', 'away_qb_id', 'away_qb_name', 'total_line', 'temp', 'wind']]
    away_players.columns = ['game_id', 'season', 'week', 'team', 'opponent', 'player_id', 'player_name', 'line', 'temp', 'wind']


    model_spine = pd.concat([home_players, away_players], ignore_index=True)

    
    df = model_spine.merge(player_stats.rename(columns={'opponent_team': 'opponent'}), how='left', on=['season', 'week', 'team', 'opponent', 'player_id'])
    df = df.merge(next_gen.rename(columns={'player_gsis_id': 'player_id'}), how='left', on=['season', 'week', 'player_id'])
    df = df.merge(team_stats, how='left', on=['season', 'week', 'team'])
    df = df.sort_values(by=['player_id', 'season', 'week'])

    targets = ['passing_yards', 'passing_tds', 'passing_interceptions', 'fumbles_lost', 'rushing_yards', 'rushing_tds']

    numeric_exclude = ['season', 'week', 'wind', 'temp', 'line']
    numeric = df.select_dtypes(include='number').drop(columns=numeric_exclude)

    for window in windows:
        rolling_df = df.groupby('player_id')[numeric.columns].transform(lambda x: x.rolling(window=window, min_periods=1).mean()).add_suffix(f'_last_{window}').shift(1)
        df = pd.concat([df, rolling_df], axis=1)
    
    df = df.drop(
        df[(df['season'] < 2025) & (df[targets].isnull().any(axis=1))].index 
    )

    df.drop(columns=['player_display_name', 'opponent_team'], inplace=True)
    df.rename(columns={'player_name': 'player_display_name'}, inplace=True)
    df = df[df['season'] == 2025]
    
    return df